import { apiRequest } from './client';

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
  filters: Record<string, FilterNode>;
}

export type FilterNode = FilterInfo | FilterGroupInfo;
export type FilterTree = Record<string, FilterNode>;

export type FilterResource = 'filters' | 'groups';

export interface CreateFilterRequest {
  expression: string;
  name?: string;
  path?: string[];
}

export interface CreateGroupRequest {
  name: string;
  path?: string[];
}

export interface CoverageResponse {
  /**
   * Weighted coverage relative to the active street's base range.
   * Filter creation returns the new filter's membership coverage;
   * activation, deletion, and clearing return effective range coverage.
   */
  coverage: number;
}

export interface GroupResponse {
  name: string;
  path: string[];
}

function baseUrl(sessionId: string, playerId: number): string {
  return `/api/sessions/${sessionId}/players/${playerId}`;
}

/**
 * Build a URL identifying a filter or group in the hierarchy.
 *
 * The path contains parent group names, excluding the target node.
 * Nested paths are encoded as repeated query parameters.
 *
 * @param resource - URL resource segment ("filters" or "groups").
 * @param name - Name of the target filter or group.
 * @param path - Ordered parent group names, from root to parent.
 */
function nodeUrl(
  sessionId: string,
  playerId: number,
  resource: FilterResource,
  name: string,
  path: string[] = [],
): string {
  const url = `${baseUrl(sessionId, playerId)}/${resource}/${encodeURIComponent(name)}`;

  if (path.length === 0) return url;

  const params = new URLSearchParams();
  path.forEach((group) => params.append('path', group));

  return `${url}?${params.toString()}`;
}

/**
 * Retrieve the complete filter hierarchy for a player's active street.
 *
 * @returns A tree keyed by root-level filter and group names.
 */
export function getFilters(sessionId: string, playerId: number): Promise<FilterTree> {
  return apiRequest<FilterTree>(`${baseUrl(sessionId, playerId)}/filters`);
}

/**
 * Retrieve a saved filter or group by name and parent path.
 *
 * Group responses include recursively nested descendants.
 *
 * @param resource - Filter or group URL.
 * @param name - Target node name.
 * @param path - Parent group hierarchy, or an empty array for root nodes.
 * @returns Metadata for the addressed filter or group.
 */
export function getFilter(
  sessionId: string,
  playerId: number,
  resource: FilterResource,
  name: string,
  path: string[] = [],
): Promise<FilterNode> {
  return apiRequest<FilterNode>(nodeUrl(sessionId, playerId, resource, name, path));
}

/**
 * Create a filter in the player's active street.
 *
 * The backend evaluates the expression against the street's base
 * range and saves its selection in the specified parent group.
 *
 * @param request - Filter expression, optional name, and parent path.
 * @returns The new filter's weighted membership coverage.
 */
export function createFilter(
  sessionId: string,
  playerId: number,
  request: CreateFilterRequest,
): Promise<CoverageResponse> {
  return apiRequest<CoverageResponse>(`${baseUrl(sessionId, playerId)}/filters`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

/**
 * Create an empty filter group at the specified parent path.
 *
 * New groups start inactive, including those created inside
 * active groups.
 *
 * @returns The created group's name and parent path.
 */
export function createGroup(
  sessionId: string,
  playerId: number,
  request: CreateGroupRequest,
): Promise<GroupResponse> {
  return apiRequest<GroupResponse>(`${baseUrl(sessionId, playerId)}/groups`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

/**
 * Activate or deactivate a saved filter or group.
 *
 * Activating a filter also activates its ancestor groups.
 * Activating or deactivating a group recursively updates
 * its descendants.
 *
 * @param resource - Filter or group URL.
 * @param name - Target node name.
 * @param active - Desired activation state.
 * @param path - Parent group hierarchy.
 * @returns The player's resulting effective weighted coverage.
 */
export function setFilterActive(
  sessionId: string,
  playerId: number,
  resource: FilterResource,
  name: string,
  active: boolean,
  path: string[] = [],
): Promise<CoverageResponse> {
  return apiRequest<CoverageResponse>(nodeUrl(sessionId, playerId, resource, name, path), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ active }),
  });
}

/**
 * Delete a saved filter or group from the active street.
 *
 * Deleting a group also removes all its descendants.
 *
 * @param resource - Filter or group URL.
 * @param name - Target node name.
 * @param path - Parent group hierarchy.
 * @returns The player's resulting effective weighted coverage.
 */
export function deleteFilter(
  sessionId: string,
  playerId: number,
  resource: FilterResource,
  name: string,
  path: string[] = [],
): Promise<CoverageResponse> {
  return apiRequest<CoverageResponse>(nodeUrl(sessionId, playerId, resource, name, path), {
    method: 'DELETE',
  });
}

/**
 * Deactivate all filters and groups in the player's active street.
 *
 * Preserves their definitions and hierarchy while restoring
 * the effective range to the street's unfiltered base range.
 *
 * @returns Effective weighted coverage of 1.0 on success.
 */
export function clearFilters(sessionId: string, playerId: number): Promise<CoverageResponse> {
  return apiRequest<CoverageResponse>(`${baseUrl(sessionId, playerId)}/filters/clear`, {
    method: 'POST',
  });
}
