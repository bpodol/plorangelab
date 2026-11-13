export type Mode = 4 | 5;

export interface FilterInfo {
  type: 'filter';
  name: string;
  expression: string;
  coverage: number;
  active: boolean;
}

/**
 * Represent a saved filter group with recursively nested children.
 *
 * Children are keyed by name and may be individual filters
 * or other groups.
 */
export interface FilterGroupInfo {
  type: 'group';
  name: string;
  coverage: number;
  active: boolean;
  filters: FilterTree;
}

export type FilterNode = FilterInfo | FilterGroupInfo;
export type FilterTree = Record<string, FilterNode>;

export interface PlayerFilterSnapshot {
  base_hands: number;
  filtered_hands: number;
  filtered_coverage: number;
  filters: FilterTree;
}
