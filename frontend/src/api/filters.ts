import { apiRequest } from './client';

export interface FilterInfo {
  type: 'filter';
  name: string;
  expression: string;
  coverage: number;
  active: boolean;
}

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
  coverage: number;
}

export interface GroupResponse {
  name: string;
  path: string[];
}

function baseUrl(sessionId: string, playerId: number): string {
  return `/api/sessions/${sessionId}/players/${playerId}`;
}

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

export function getFilters(sessionId: string, playerId: number): Promise<FilterTree> {
  return apiRequest<FilterTree>(`${baseUrl(sessionId, playerId)}/filters`);
}

export function getFilter(
  sessionId: string,
  playerId: number,
  resource: FilterResource,
  name: string,
  path: string[] = [],
): Promise<FilterNode> {
  return apiRequest<FilterNode>(nodeUrl(sessionId, playerId, resource, name, path));
}

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

export function clearFilters(sessionId: string, playerId: number): Promise<CoverageResponse> {
  return apiRequest<CoverageResponse>(`${baseUrl(sessionId, playerId)}/filters/clear`, {
    method: 'POST',
  });
}
