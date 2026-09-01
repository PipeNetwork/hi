package model

type User struct {
	ID          int    `json:"user_id,omitempty"`
	DisplayName string `json:"displayName"`
	Email       string `json:"email"`
}
