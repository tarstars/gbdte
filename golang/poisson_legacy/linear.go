package poissonlegacy

import (
	"gonum.org/v1/gonum/mat"
)

func solveLinearSystem(hess [][]float64, grad []float64, regLambda float64) ([]float64, error) {
	d := len(hess)
	lhs := mat.NewDense(d, d, nil)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			val := hess[i][j]
			if i == j {
				val += regLambda
			}
			lhs.Set(i, j, val)
		}
	}
	rhs := mat.NewDense(d, 1, nil)
	for i := 0; i < d; i++ {
		rhs.Set(i, 0, grad[i])
	}
	var out mat.Dense
	if err := out.Solve(lhs, rhs); err != nil {
		return nil, err
	}
	result := make([]float64, d)
	for i := 0; i < d; i++ {
		result[i] = out.At(i, 0)
	}
	return result, nil
}
