export interface User {
  id: string;
  name: string;
}

export async function fetchUser(id: number): Promise<User> {
  const response = await fetch(`/users/${id}`);
  return response.json() as Promise<User>;
}
