import { useState, type SubmitEvent } from 'react';
import type { FilterResource } from '../api/filters';
import type { FilterTree, PlayerFilterSnapshot } from '../types';

type AddFilter = (
  playerId: number,
  expression: string,
  name?: string,
  path?: string[],
) => Promise<boolean>;

type AddGroup = (playerId: number, name: string, path?: string[]) => Promise<boolean>;

type NodeAction = (
  playerId: number,
  resource: FilterResource,
  name: string,
  path?: string[],
) => Promise<boolean>;

interface FilterPanelProps {
  playerId: number;
  filterSnapshot: PlayerFilterSnapshot | null;
  disabled?: boolean;
  onAddFilter: AddFilter;
  onAddGroup: AddGroup;
  onToggleFilterActive: NodeAction;
  onRemoveFilter?: NodeAction;
  onClearAllFilters?: (playerId: number) => Promise<boolean>;
}

interface AddNodeControlProps {
  playerId: number;
  path: string[];
  disabled: boolean;
  onAddFilter: AddFilter;
  onAddGroup: AddGroup;
}

const percent = new Intl.NumberFormat(undefined, {
  style: 'percent',
  maximumFractionDigits: 2,
});

const handCount = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
});

/**
 * Render a reusable form for creating a filter or filter group.
 *
 * Displays input fields based on the selected node type and delegates
 * creation to the provided callbacks using the parent group's path.
 *
 * Maintains local form state, including input values, submission
 * status, and errors.
 *
 * @param playerId - Zero-based player slot index (0-5).
 * @param path - Parent group hierarchy, or an empty array for the root.
 * @param disabled - Whether creation controls are disabled.
 * @param onAddFilter - Callback for creating a filter.
 * @param onAddGroup - Callback for creating a group.
 */
function AddNodeControl({
  playerId,
  path,
  disabled,
  onAddFilter,
  onAddGroup,
}: AddNodeControlProps) {
  const [open, setOpen] = useState(false);
  const [type, setType] = useState<'filter' | 'group'>('filter');
  const [name, setName] = useState('');
  const [expression, setExpression] = useState('');
  const [submitting, setSubmitting] = useState(false);

  function reset() {
    setOpen(false);
    setType('filter');
    setName('');
    setExpression('');
  }

  /**
   * Handle submission of the filter or group creation form.
   *
   * Prevents the default form submission and delegates creation to
   * the appropriate callback based on the selected node type.
   *
   * Closes and resets the form only when the operation succeeds.
   * On failure, preserves the inputs so the user can retry.
   *
   * @param event - React form submission event.
   */
  async function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled || submitting) return;

    setSubmitting(true);

    try {
      let success: boolean;

      if (type === 'filter') {
        success = await onAddFilter(playerId, expression.trim(), name.trim() || undefined, path);
      } else {
        success = await onAddGroup(playerId, name.trim(), path);
      }
      if (success) reset();
    } finally {
      setSubmitting(false);
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        className="filter-panel__button"
        onClick={() => setOpen(true)}
        disabled={disabled}
        aria-label={`Add filter or group ${path.length ? `inside ${path.join(' / ')}` : 'at root'}`}
      >
        + Add
      </button>
    );
  }

  return (
    <form className="filter-panel__add-form" onSubmit={(event) => void handleSubmit(event)}>
      <label>
        Type
        <select
          value={type}
          onChange={(event) => setType(event.target.value as 'filter' | 'group')}
          disabled={disabled || submitting}
        >
          <option value="filter">Filter</option>
          <option value="group">Group</option>
        </select>
      </label>

      <label>
        {type === 'filter' ? 'Name (optional)' : 'Group name'}
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          required={type === 'group'}
          disabled={disabled || submitting}
          placeholder={type === 'filter' ? 'Defaults to expression' : 'e.g. Draws'}
        />
      </label>

      {type === 'filter' && (
        <label>
          Filter expression
          <input
            value={expression}
            onChange={(event) => setExpression(event.target.value)}
            required
            disabled={disabled || submitting}
            placeholder="Enter a filter expression"
          />
        </label>
      )}

      <div className="filter-panel__actions">
        <button
          type="submit"
          className="filter-panel__button filter-panel__button--primary"
          disabled={
            disabled || submitting || (type === 'filter' ? !expression.trim() : !name.trim())
          }
        >
          {submitting ? 'Adding…' : `Add ${type}`}
        </button>
        <button
          type="button"
          className="filter-panel__button"
          onClick={reset}
          disabled={submitting}
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

interface TreeProps {
  tree: FilterTree;
  path: string[];
  playerId: number;
  disabled: boolean;
  onAddFilter: AddFilter;
  onAddGroup: AddGroup;
  onToggleFilterActive: NodeAction;
  onRemoveFilter?: NodeAction;
}

/**
 * Recursively render a filter tree and its nested groups.
 *
 * Displays each filter or group's name, coverage, activation checkbox,
 * and optional deletion control.
 *
 * Each group renders its children recursively and provides an add
 * control with the path extended by the group's name.
 *
 * @param props - Filter tree, current parent path, player ID,
 * disabled state, and mutation callbacks.
 */
function FilterTreeView(props: TreeProps) {
  const {
    tree,
    path,
    playerId,
    disabled,
    onAddFilter,
    onAddGroup,
    onToggleFilterActive,
    onRemoveFilter,
  } = props;

  if (Object.keys(tree).length === 0) {
    return <p className="filter-panel__empty">No filters in this group.</p>;
  }

  return (
    <ul className="filter-panel__tree">
      {Object.entries(tree).map(([name, node]) => {
        const resource: FilterResource = node.type === 'filter' ? 'filters' : 'groups';
        const childPath = [...path, name];

        return (
          <li className="filter-panel__item" key={name}>
            <div className="filter-panel__node">
              <div className="filter-panel__node-details">
                <span className="filter-panel__kind">{node.type} </span>
                <span className="filter-panel__name">{node.name} </span>
                {/* {node.type === 'filter' && (
                  <span className="filter-panel__expression" title={node.expression}>
                    {node.expression}
                  </span>
                )} */}
                <span className="filter-panel__coverage">{percent.format(node.coverage)}</span>
              </div>

              <div className="filter-panel__actions">
                <input
                  type="checkbox"
                  checked={node.active}
                  onChange={() => void onToggleFilterActive(playerId, resource, name, path)}
                  disabled={disabled}
                  aria-label={`Activate ${node.type} ${node.name}`}
                />

                {node.type === 'group' && (
                  <AddNodeControl
                    playerId={playerId}
                    path={childPath}
                    disabled={disabled}
                    onAddFilter={onAddFilter}
                    onAddGroup={onAddGroup}
                  />
                )}

                {onRemoveFilter && (
                  <button
                    type="button"
                    className="filter-panel__button filter-panel__button--danger"
                    title={
                      node.type === 'group'
                        ? 'Delete group and all its descendants'
                        : 'Delete filter'
                    }
                    aria-label={`Delete ${node.type} ${node.name}`}
                    onClick={() => {
                      if (
                        node.type === 'group' &&
                        Object.keys(node.filters).length > 0 &&
                        !window.confirm(`Delete group "${node.name}" and all its descendants?`)
                      )
                        return;
                      void onRemoveFilter(playerId, resource, name, path);
                    }}
                    disabled={disabled}
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>

            {node.type === 'group' && (
              <div className="filter-panel__children">
                <FilterTreeView {...props} tree={node.filters} path={childPath} />
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Display a player's effective hand count, coverage, and nested filter tree.
 *
 * Renders root and group-specific add controls. Each group supplies its own
 * address path to the shared creation form; the caller never enters paths.
 * All mutations are delegated to usePlayerRanges callbacks.
 */
export function FilterPanel({
  playerId,
  filterSnapshot,
  disabled = false,
  onAddFilter,
  onAddGroup,
  onToggleFilterActive,
  onRemoveFilter,
  onClearAllFilters,
}: FilterPanelProps) {
  const filters = filterSnapshot?.filters ?? {};
  const hasFilters = Object.keys(filters).length > 0;

  return (
    <section className="filter-panel" aria-label={`Player ${playerId + 1} filters`}>
      <div className="filter-panel__header">
        <h3>Filters — Player {playerId + 1}</h3>
        <div className="filter-panel__actions">
          <AddNodeControl
            playerId={playerId}
            path={[]}
            disabled={disabled}
            onAddFilter={onAddFilter}
            onAddGroup={onAddGroup}
          />
          {onClearAllFilters && (
            <button
              type="button"
              className="filter-panel__button"
              onClick={() => void onClearAllFilters(playerId)}
              disabled={disabled || !hasFilters}
            >
              Deactivate all
            </button>
          )}
        </div>
      </div>

      <div className="filter-panel__stats">
        <span>
          Filtered hands:{' '}
          <strong>{filterSnapshot ? handCount.format(filterSnapshot.filtered_hands) : '—'}</strong>
        </span>
        <span>
          Filtered coverage:{' '}
          <strong>{filterSnapshot ? percent.format(filterSnapshot.filtered_coverage) : '—'}</strong>
        </span>
      </div>

      {hasFilters ? (
        <FilterTreeView
          tree={filters}
          path={[]}
          playerId={playerId}
          disabled={disabled}
          onAddFilter={onAddFilter}
          onAddGroup={onAddGroup}
          onToggleFilterActive={onToggleFilterActive}
          onRemoveFilter={onRemoveFilter}
        />
      ) : (
        <p className="filter-panel__empty">
          {filterSnapshot ? 'No filters yet.' : 'No filter snapshot available yet.'}
        </p>
      )}
    </section>
  );
}
