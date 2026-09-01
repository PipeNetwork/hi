export interface User {
  id: string;
  name: string;
}

export async function loadUser(id: string): Promise<User> {
  const response = await fetch(`/users/${encodeURIComponent(id)}`);
  return response.json() as Promise<User>;
}
