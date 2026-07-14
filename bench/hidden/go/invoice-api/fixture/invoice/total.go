package invoice

func Total(items []LineItem) float64 {
	var total float64
	for _, item := range items {
		total += item.UnitPrice * float64(item.Quantity)
	}
	return total
}
