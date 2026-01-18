package poissonlegacy

import (
	"errors"
	"math"
	"sort"
)

type TimeFunc func(float64) float64

func Piecewise(a, b float64) TimeFunc {
	return func(x float64) float64 {
		if x >= a && x < b {
			return 1.0
		}
		return 0.0
	}
}

func make2D(rows, cols int) [][]float64 {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return data
}

func make2DInt(rows, cols int) [][]int {
	data := make([][]int, rows)
	for i := range data {
		data[i] = make([]int, cols)
	}
	return data
}

func make3D(rows, cols, depth int) [][][]float64 {
	data := make([][][]float64, rows)
	for i := range data {
		data[i] = make([][]float64, cols)
		for j := range data[i] {
			data[i][j] = make([]float64, depth)
		}
	}
	return data
}

func make4D(rows, cols, depth int) [][][][]float64 {
	data := make([][][][]float64, rows)
	for i := range data {
		data[i] = make([][][]float64, cols)
		for j := range data[i] {
			data[i][j] = make([][]float64, depth)
			for k := range data[i][j] {
				data[i][j][k] = make([]float64, depth)
			}
		}
	}
	return data
}

func edgePenalty(n int) []float64 {
	if n <= 1 {
		return nil
	}
	penalty := make([]float64, n-1)
	center := float64(n) / 2.0
	for i := 0; i < n-1; i++ {
		penalty[i] = math.Abs(float64(i) - center)
	}
	return penalty
}

func makeGax(features [][]float64) ([][]int, error) {
	if len(features) == 0 {
		return nil, errors.New("empty features")
	}
	cols := len(features[0])
	for i := range features {
		if len(features[i]) != cols {
			return nil, errors.New("inconsistent feature widths")
		}
	}

	rows := len(features)
	gax := make2DInt(rows, cols)
	indices := make([]int, rows)
	for col := 0; col < cols; col++ {
		for i := range indices {
			indices[i] = i
		}
		sort.SliceStable(indices, func(i, j int) bool {
			ai := features[indices[i]][col]
			aj := features[indices[j]][col]
			if ai == aj {
				return indices[i] < indices[j]
			}
			return ai < aj
		})
		for row := 0; row < rows; row++ {
			gax[row][col] = indices[row]
		}
	}
	return gax, nil
}

func takeAlongAxis(features [][]float64, gax [][]int) [][]float64 {
	rows := len(gax)
	if rows == 0 {
		return nil
	}
	cols := len(gax[0])
	out := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[i][j] = features[gax[i][j]][j]
		}
	}
	return out
}

func gatherVectorByGax(values []float64, gax [][]int) [][]float64 {
	rows := len(gax)
	if rows == 0 {
		return nil
	}
	cols := len(gax[0])
	out := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[i][j] = values[gax[i][j]]
		}
	}
	return out
}

func gatherIntByGax(values []int, gax [][]int) [][]int {
	rows := len(gax)
	if rows == 0 {
		return nil
	}
	cols := len(gax[0])
	out := make2DInt(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[i][j] = values[gax[i][j]]
		}
	}
	return out
}

func gatherExtraByGax(values [][]float64, gax [][]int) [][][]float64 {
	rows := len(gax)
	if rows == 0 {
		return nil
	}
	cols := len(gax[0])
	depth := len(values[0])
	out := make3D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			row := values[gax[i][j]]
			copy(out[i][j], row)
		}
	}
	return out
}

func cumsumForward2D(data [][]float64) [][]float64 {
	rows := len(data)
	if rows == 0 {
		return nil
	}
	cols := len(data[0])
	out := make2D(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := data[i][j]
			if i > 0 {
				val += out[i-1][j]
			}
			out[i][j] = val
		}
	}
	return out
}

func cumsumBackward2D(data [][]float64) [][]float64 {
	rows := len(data)
	if rows == 0 {
		return nil
	}
	cols := len(data[0])
	out := make2D(rows, cols)
	for i := rows - 1; i >= 0; i-- {
		for j := 0; j < cols; j++ {
			val := data[i][j]
			if i < rows-1 {
				val += out[i+1][j]
			}
			out[i][j] = val
		}
	}
	return out
}

func cumsumForward3D(data [][][]float64) [][][]float64 {
	rows := len(data)
	if rows == 0 {
		return nil
	}
	cols := len(data[0])
	depth := len(data[0][0])
	out := make3D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for k := 0; k < depth; k++ {
				val := data[i][j][k]
				if i > 0 {
					val += out[i-1][j][k]
				}
				out[i][j][k] = val
			}
		}
	}
	return out
}

func cumsumBackward3D(data [][][]float64) [][][]float64 {
	rows := len(data)
	if rows == 0 {
		return nil
	}
	cols := len(data[0])
	depth := len(data[0][0])
	out := make3D(rows, cols, depth)
	for i := rows - 1; i >= 0; i-- {
		for j := 0; j < cols; j++ {
			for k := 0; k < depth; k++ {
				val := data[i][j][k]
				if i < rows-1 {
					val += out[i+1][j][k]
				}
				out[i][j][k] = val
			}
		}
	}
	return out
}

func cumsumForward4D(data [][][][]float64) [][][][]float64 {
	rows := len(data)
	if rows == 0 {
		return nil
	}
	cols := len(data[0])
	depth := len(data[0][0])
	out := make4D(rows, cols, depth)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for k := 0; k < depth; k++ {
				for l := 0; l < depth; l++ {
					val := data[i][j][k][l]
					if i > 0 {
						val += out[i-1][j][k][l]
					}
					out[i][j][k][l] = val
				}
			}
		}
	}
	return out
}

func cumsumBackward4D(data [][][][]float64) [][][][]float64 {
	rows := len(data)
	if rows == 0 {
		return nil
	}
	cols := len(data[0])
	depth := len(data[0][0])
	out := make4D(rows, cols, depth)
	for i := rows - 1; i >= 0; i-- {
		for j := 0; j < cols; j++ {
			for k := 0; k < depth; k++ {
				for l := 0; l < depth; l++ {
					val := data[i][j][k][l]
					if i < rows-1 {
						val += out[i+1][j][k][l]
					}
					out[i][j][k][l] = val
				}
			}
		}
	}
	return out
}

func countObjects(bjids []int, gax [][]int) [][]float64 {
	rows := len(gax)
	if rows == 0 {
		return nil
	}
	cols := len(gax[0])
	bjidRect := gatherIntByGax(bjids, gax)
	counts := make2D(rows, cols)
	for j := 0; j < cols; j++ {
		counts[0][j] = 1
		for i := 1; i < rows; i++ {
			inc := 0.0
			if bjidRect[i][j] != bjidRect[i-1][j] {
				inc = 1.0
			}
			counts[i][j] = counts[i-1][j] + inc
		}
	}
	return counts
}

func reverseGax(gax [][]int) [][]int {
	rows := len(gax)
	if rows == 0 {
		return nil
	}
	cols := len(gax[0])
	out := make2DInt(rows, cols)
	for i := 0; i < rows; i++ {
		copy(out[i], gax[rows-1-i])
	}
	return out
}

func smartThreshold(sortedValues []float64, idx int) (float64, bool) {
	if len(sortedValues) < 2 {
		return 0, false
	}
	uniq := make([]float64, 0, len(sortedValues))
	indices := make([]int, len(sortedValues))
	for i, v := range sortedValues {
		if i == 0 || v != uniq[len(uniq)-1] {
			uniq = append(uniq, v)
		}
		indices[i] = len(uniq) - 1
	}
	if len(uniq) < 2 {
		return 0, false
	}
	if idx < 0 {
		idx = 0
	}
	if idx >= len(indices) {
		idx = len(indices) - 1
	}
	indInUnique := indices[idx]
	if indInUnique > len(uniq)-2 {
		indInUnique = len(uniq) - 2
	}
	return (uniq[indInUnique] + uniq[indInUnique+1]) / 2.0, true
}

func sliceFloat(values []float64, mask []bool) []float64 {
	out := make([]float64, 0, len(values))
	for i, v := range values {
		if mask[i] {
			out = append(out, v)
		}
	}
	return out
}

func sliceInt(values []int, mask []bool) []int {
	out := make([]int, 0, len(values))
	for i, v := range values {
		if mask[i] {
			out = append(out, v)
		}
	}
	return out
}

func slice2D(values [][]float64, mask []bool) [][]float64 {
	out := make([][]float64, 0, len(values))
	for i, row := range values {
		if mask[i] {
			cloned := make([]float64, len(row))
			copy(cloned, row)
			out = append(out, cloned)
		}
	}
	return out
}

func sliceOptional2D(values [][]float64, mask []bool) [][]float64 {
	if values == nil {
		return nil
	}
	return slice2D(values, mask)
}

func sliceOptionalFloat(values []float64, mask []bool) []float64 {
	if values == nil {
		return nil
	}
	return sliceFloat(values, mask)
}
