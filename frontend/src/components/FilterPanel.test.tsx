import type { ComponentProps } from 'react';
import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, test, vi } from 'vitest';

import { FilterPanel } from './FilterPanel';
import type { PlayerFilterSnapshot } from '../types';

const snapshot: PlayerFilterSnapshot = {
  base_hands: 1000,
  filtered_hands: 125.5,
  filtered_coverage: 0.1255,
  filters: {
    'top-pair': {
      type: 'filter',
      name: 'top-pair',
      expression: 'TP',
      coverage: 0.25,
      active: false,
    },
    Draws: {
      type: 'group',
      name: 'Draws',
      coverage: 0.4,
      active: true,
      filters: {
        'nut-fd': {
          type: 'filter',
          name: 'nut-fd',
          expression: 'NFD',
          coverage: 0.2,
          active: true,
        },
        Backdoors: {
          type: 'group',
          name: 'Backdoors',
          coverage: 0.08,
          active: true,
          filters: {
            bdfd: {
              type: 'filter',
              name: 'bdfd',
              expression: 'BDFD',
              coverage: 0.0625,
              active: false,
            },
          },
        },
      },
    },
  },
};

type Props = ComponentProps<typeof FilterPanel>;

function renderPanel(overrides: Partial<Props> = {}) {
  const props = {
    playerId: 2,
    filterSnapshot: snapshot,
    onAddFilter: vi.fn().mockResolvedValue(true),
    onAddGroup: vi.fn().mockResolvedValue(true),
    onToggleFilterActive: vi.fn().mockResolvedValue(true),
    onRemoveFilter: vi.fn().mockResolvedValue(true),
    onClearAllFilters: vi.fn().mockResolvedValue(true),
    ...overrides,
  };

  const view = render(<FilterPanel {...props} />);
  return { ...view, props };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('FilterPanel', () => {
  test('renders the supplied stats and reveals the nested tree when expanded', async () => {
    const user = userEvent.setup();
    renderPanel();

    const panel = screen.getByRole('region', { name: 'Player 3 filters' });
    const formatPercent = new Intl.NumberFormat(undefined, {
      style: 'percent',
      maximumFractionDigits: 2,
    });
    const formatHands = new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 2,
    });

    expect(
      within(panel).getByText(formatHands.format(125.5), { selector: 'strong' }),
    ).toBeInTheDocument();
    expect(
      within(panel).getByText(formatPercent.format(0.1255), { selector: 'strong' }),
    ).toBeInTheDocument();

    expect(within(panel).queryByText('nut-fd')).not.toBeInTheDocument();
    expect(within(panel).getByRole('button', { name: 'Draws' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );

    await user.click(within(panel).getByRole('button', { name: 'Draws' }));
    await user.click(within(panel).getByRole('button', { name: 'Backdoors' }));

    for (const nodeName of ['top-pair', 'Draws', 'nut-fd', 'Backdoors', 'bdfd']) {
      expect(within(panel).getByText(nodeName)).toBeInTheDocument();
    }

    for (const coverage of [0.25, 0.4, 0.2, 0.08, 0.0625]) {
      expect(
        within(panel).getByText(formatPercent.format(coverage), {
          selector: '.filter-panel__coverage',
        }),
      ).toBeInTheDocument();
    }

    expect(
      within(panel).getByRole('checkbox', { name: 'Activate filter top-pair' }),
    ).not.toBeChecked();
    expect(within(panel).getByRole('checkbox', { name: 'Activate group Draws' })).toBeChecked();
    expect(within(panel).getByRole('checkbox', { name: 'Activate filter nut-fd' })).toBeChecked();
    expect(within(panel).getByRole('checkbox', { name: 'Activate filter bdfd' })).not.toBeChecked();
  });

  test('collapsing a parent unmounts nested groups and resets their expansion state', async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole('button', { name: 'Draws' }));
    await user.click(screen.getByRole('button', { name: 'Backdoors' }));
    expect(screen.getByText('bdfd')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Draws' }));
    expect(screen.queryByText('Backdoors')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Draws' }));
    expect(screen.getByRole('button', { name: 'Backdoors' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    expect(screen.queryByText('bdfd')).not.toBeInTheDocument();
  });

  test('distinguishes a missing snapshot from an empty filter tree', () => {
    const { rerender, props } = renderPanel({ filterSnapshot: null });

    expect(screen.getByText('No filter snapshot available yet.')).toBeInTheDocument();
    expect(screen.getAllByText('—')).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Deactivate all' })).toBeDisabled();

    rerender(
      <FilterPanel
        {...props}
        filterSnapshot={{
          base_hands: 100,
          filtered_hands: 100,
          filtered_coverage: 1,
          filters: {},
        }}
      />,
    );

    expect(screen.getByText('No filters yet.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Deactivate all' })).toBeDisabled();
  });

  test('submits a root filter with trimmed values and an empty parent path', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Add filter at root' }));
    await user.type(screen.getByRole('textbox', { name: 'Filter expression' }), '  AA  ');
    await user.type(screen.getByRole('textbox', { name: 'Name (optional)' }), '  Aces  ');
    await user.click(screen.getByRole('button', { name: 'Add filter' }));

    expect(props.onAddFilter).toHaveBeenCalledTimes(1);
    expect(props.onAddFilter).toHaveBeenCalledWith(2, 'AA', 'Aces', []);
    expect(props.onAddGroup).not.toHaveBeenCalled();
    expect(await screen.findByRole('button', { name: 'Add filter at root' })).toBeInTheDocument();
  });

  test('uses undefined when an optional filter name is left blank', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Add filter at root' }));
    await user.type(screen.getByRole('textbox', { name: 'Filter expression' }), 'NFD');
    await user.click(screen.getByRole('button', { name: 'Add filter' }));

    expect(props.onAddFilter).toHaveBeenCalledTimes(1);
    expect(props.onAddFilter).toHaveBeenCalledWith(2, 'NFD', undefined, []);
  });

  test('opens the group form and creates a group at the root', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Add group at root' }));

    expect(screen.queryByRole('textbox', { name: 'Filter expression' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add group' })).toBeDisabled();

    await user.type(screen.getByRole('textbox', { name: 'Group name' }), '  Made hands  ');
    await user.click(screen.getByRole('button', { name: 'Add group' }));

    expect(props.onAddGroup).toHaveBeenCalledTimes(1);
    expect(props.onAddGroup).toHaveBeenCalledWith(2, 'Made hands', []);
    expect(props.onAddFilter).not.toHaveBeenCalled();
  });

  test('derives the full parent path when adding inside a nested group', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Draws' }));
    await user.click(screen.getByRole('button', { name: 'Backdoors' }));
    await user.click(screen.getByRole('button', { name: 'Add filter inside Draws / Backdoors' }));
    await user.type(screen.getByRole('textbox', { name: 'Filter expression' }), 'BDSD');
    await user.click(screen.getByRole('button', { name: 'Add filter' }));

    expect(props.onAddFilter).toHaveBeenCalledTimes(1);
    expect(props.onAddFilter).toHaveBeenCalledWith(2, 'BDSD', undefined, ['Draws', 'Backdoors']);
  });

  test('keeps the creation form and entered values when the callback returns false', async () => {
    const user = userEvent.setup();
    const onAddFilter = vi.fn().mockResolvedValue(false);
    renderPanel({ onAddFilter });

    await user.click(screen.getByRole('button', { name: 'Add filter at root' }));
    await user.type(screen.getByRole('textbox', { name: 'Filter expression' }), 'INVALID');
    await user.click(screen.getByRole('button', { name: 'Add filter' }));

    expect(onAddFilter).toHaveBeenCalledOnce();
    expect(screen.getByRole('textbox', { name: 'Filter expression' })).toHaveValue('INVALID');
    expect(screen.getByRole('button', { name: 'Add filter' })).toBeEnabled();
    // The hook owns errors; the form should not render a duplicate error message.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  test('prevents duplicate submissions while creation is pending', async () => {
    const user = userEvent.setup();
    let resolveRequest!: (success: boolean) => void;
    const pending = new Promise<boolean>((resolve) => {
      resolveRequest = resolve;
    });
    const onAddFilter = vi.fn().mockReturnValue(pending);
    renderPanel({ onAddFilter });

    await user.click(screen.getByRole('button', { name: 'Add filter at root' }));
    await user.type(screen.getByRole('textbox', { name: 'Filter expression' }), 'AA');
    await user.click(screen.getByRole('button', { name: 'Add filter' }));

    expect(onAddFilter).toHaveBeenCalledOnce();
    expect(screen.getByRole('button', { name: 'Adding…' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();

    await act(async () => {
      resolveRequest(true);
      await pending;
    });

    expect(screen.getByRole('button', { name: 'Add filter at root' })).toBeInTheDocument();
  });

  test('cancel discards draft input and resets the selected node type', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Add group at root' }));
    await user.type(screen.getByRole('textbox', { name: 'Group name' }), 'Draft');
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: 'Add filter at root' }));

    expect(screen.queryByRole('textbox', { name: 'Group name' })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Name (optional)' })).toHaveValue('');
    expect(screen.getByRole('textbox', { name: 'Filter expression' })).toHaveValue('');
    expect(props.onAddGroup).not.toHaveBeenCalled();
  });

  test('forwards toggle actions for both filters and nested groups', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Draws' }));
    await user.click(screen.getByRole('button', { name: 'Backdoors' }));
    await user.click(screen.getByRole('checkbox', { name: 'Activate filter top-pair' }));
    await user.click(screen.getByRole('checkbox', { name: 'Activate group Backdoors' }));
    await user.click(screen.getByRole('checkbox', { name: 'Activate filter bdfd' }));

    expect(props.onToggleFilterActive).toHaveBeenNthCalledWith(1, 2, 'filters', 'top-pair', []);
    expect(props.onToggleFilterActive).toHaveBeenNthCalledWith(2, 2, 'groups', 'Backdoors', [
      'Draws',
    ]);
    expect(props.onToggleFilterActive).toHaveBeenNthCalledWith(3, 2, 'filters', 'bdfd', [
      'Draws',
      'Backdoors',
    ]);
    // The checkbox remains controlled by the snapshot prop, not local UI state.
    expect(screen.getByRole('checkbox', { name: 'Activate filter top-pair' })).not.toBeChecked();
  });

  test('forwards deletion for a nested filter without confirmation', async () => {
    const user = userEvent.setup();
    const confirm = vi.spyOn(window, 'confirm');
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Draws' }));
    await user.click(screen.getByRole('button', { name: 'Backdoors' }));
    await user.click(screen.getByRole('button', { name: 'Delete filter bdfd' }));

    expect(confirm).not.toHaveBeenCalled();
    expect(props.onRemoveFilter).toHaveBeenCalledTimes(1);
    expect(props.onRemoveFilter).toHaveBeenCalledWith(2, 'filters', 'bdfd', ['Draws', 'Backdoors']);
  });

  test('requires confirmation before removing a nonempty group', async () => {
    const user = userEvent.setup();
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Delete group Draws' }));
    expect(confirm).toHaveBeenCalledWith('Delete group "Draws" and all its descendants?');
    expect(props.onRemoveFilter).not.toHaveBeenCalled();

    confirm.mockReturnValue(true);
    await user.click(screen.getByRole('button', { name: 'Delete group Draws' }));
    expect(props.onRemoveFilter).toHaveBeenCalledTimes(1);
    expect(props.onRemoveFilter).toHaveBeenCalledWith(2, 'groups', 'Draws', []);
  });

  test('deactivate all sends the player ID without changing the local tree', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel();

    await user.click(screen.getByRole('button', { name: 'Deactivate all' }));

    expect(props.onClearAllFilters).toHaveBeenCalledTimes(1);
    expect(props.onClearAllFilters).toHaveBeenCalledWith(2);
    expect(screen.getByRole('checkbox', { name: 'Activate group Draws' })).toBeChecked();
  });

  test('disables all mutation controls when the panel is disabled', async () => {
    const user = userEvent.setup();
    const { props } = renderPanel({ disabled: true });

    expect(screen.getByRole('button', { name: 'Add filter at root' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Add group at root' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Draws' }));
    await user.click(screen.getByRole('button', { name: 'Backdoors' }));
    expect(
      screen.getByRole('button', { name: 'Add filter inside Draws / Backdoors' }),
    ).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Deactivate all' })).toBeDisabled();

    for (const checkbox of screen.getAllByRole('checkbox')) {
      expect(checkbox).toBeDisabled();
    }
    for (const button of screen.getAllByRole('button', { name: /^Delete / })) {
      expect(button).toBeDisabled();
    }

    await user.click(screen.getByRole('button', { name: 'Add filter at root' }));
    expect(screen.queryByRole('textbox', { name: 'Filter expression' })).not.toBeInTheDocument();
    expect(props.onAddFilter).not.toHaveBeenCalled();
  });
});
