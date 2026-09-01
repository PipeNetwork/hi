package invoice

func TotalCents(items []LineItem) int64 {
	var total int64
	for _, item := range items {
		quantity := item.Quantity
		if quantity < 0 {
			quantity = 0
		}
		total += item.UnitCents * quantity
	}
	return total
}
