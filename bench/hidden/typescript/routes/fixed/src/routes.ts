export const API_PREFIX = "/api/v2";
export const projectRoute = `${API_PREFIX}/projects`;
export const userRoute = (id: string) =>
  `${API_PREFIX}/users/${encodeURIComponent(id)}`;
