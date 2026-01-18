package main

/*
#cgo CFLAGS: -I.
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"sync"
	"unsafe"

	pl "github.com/tarstars/extra_bridged_boosting/golang/poisson_legacy"
)

var (
	handleMu   sync.Mutex
	nextHandle uint64 = 1
	boosters          = make(map[uint64]*pl.Booster)

	lastErrorMu sync.Mutex
	lastError   string
)

func setLastError(err error) {
	lastErrorMu.Lock()
	defer lastErrorMu.Unlock()
	if err != nil {
		lastError = err.Error()
	} else {
		lastError = ""
	}
}

func getLastError() string {
	lastErrorMu.Lock()
	defer lastErrorMu.Unlock()
	return lastError
}

func storeBooster(b *pl.Booster) uint64 {
	handleMu.Lock()
	defer handleMu.Unlock()
	handle := nextHandle
	boosters[handle] = b
	nextHandle++
	return handle
}

func fetchBooster(handle uint64) (*pl.Booster, error) {
	handleMu.Lock()
	defer handleMu.Unlock()
	booster, ok := boosters[handle]
	if !ok {
		return nil, errors.New("invalid booster handle")
	}
	return booster, nil
}

//export PoissonFreeModel
func PoissonFreeModel(handle C.ulonglong) {
	handleMu.Lock()
	defer handleMu.Unlock()
	delete(boosters, uint64(handle))
}

func copyFloatSlice(ptr *C.double, length int) ([]float64, error) {
	if length < 0 {
		return nil, errors.New("negative length")
	}
	if length == 0 {
		return nil, nil
	}
	if ptr == nil {
		return nil, errors.New("null pointer for non-empty slice")
	}
	src := unsafe.Slice((*float64)(unsafe.Pointer(ptr)), length)
	dst := make([]float64, length)
	copy(dst, src)
	return dst, nil
}

func copyIntSlice(ptr *C.int, length int) ([]int, error) {
	if length < 0 {
		return nil, errors.New("negative length")
	}
	if length == 0 {
		return nil, nil
	}
	if ptr == nil {
		return nil, errors.New("null pointer for non-empty slice")
	}
	src := unsafe.Slice((*int32)(unsafe.Pointer(ptr)), length)
	dst := make([]int, length)
	for i, v := range src {
		dst[i] = int(v)
	}
	return dst, nil
}

func buildMatrix(ptr *C.double, rows, cols C.int) ([][]float64, error) {
	r := int(rows)
	c := int(cols)
	if r < 0 || c < 0 {
		return nil, errors.New("invalid matrix dimensions")
	}
	if r == 0 || c == 0 {
		return make([][]float64, r), nil
	}
	data, err := copyFloatSlice(ptr, r*c)
	if err != nil {
		return nil, err
	}
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, c)
		copy(row, data[i*c:(i+1)*c])
		out[i] = row
	}
	return out, nil
}

//export PoissonTrainModel
func PoissonTrainModel(
	bjidsPtr *C.int,
	rows C.int,
	freqsPtr *C.double,
	featuresInterPtr *C.double,
	interCols C.int,
	featuresExtraPtr *C.double,
	extraCols C.int,
	psiPtr *C.double,
	nStages C.int,
	maxDepth C.int,
	learningRate C.double,
	regLambda C.double,
	unbalancedPenalty C.double,
	checkZero C.int,
) C.ulonglong {
	setLastError(nil)

	if rows <= 0 {
		setLastError(errors.New("rows must be positive"))
		return 0
	}

	bjids, err := copyIntSlice(bjidsPtr, int(rows))
	if err != nil {
		setLastError(err)
		return 0
	}
	freqs, err := copyFloatSlice(freqsPtr, int(rows))
	if err != nil {
		setLastError(err)
		return 0
	}

	featuresInter, err := buildMatrix(featuresInterPtr, rows, interCols)
	if err != nil {
		setLastError(err)
		return 0
	}

	var featuresExtra [][]float64
	var psi []float64
	if extraCols > 0 {
		featuresExtra, err = buildMatrix(featuresExtraPtr, rows, extraCols)
		if err != nil {
			setLastError(err)
			return 0
		}
		psi, err = copyFloatSlice(psiPtr, int(extraCols))
		if err != nil {
			setLastError(err)
			return 0
		}
	}

	matrix, err := pl.NewPMatrixFromDense(bjids, freqs, featuresInter, featuresExtra, psi)
	if err != nil {
		setLastError(err)
		return 0
	}

	params := pl.TreeBuildParams{
		MaxDepth:          int(maxDepth),
		LearningRate:      float64(learningRate),
		UnbalancedPenalty: float64(unbalancedPenalty),
		RegLambda:         float64(regLambda),
		CheckZero:         checkZero != 0,
	}

	booster, err := pl.Train(params, matrix, int(nStages))
	if err != nil {
		setLastError(err)
		return 0
	}

	handle := storeBooster(booster)
	return C.ulonglong(handle)
}

//export PoissonPredict
func PoissonPredict(
	handle C.ulonglong,
	featuresInterPtr *C.double,
	rows C.int,
	interCols C.int,
	featuresExtraPtr *C.double,
	extraCols C.int,
	outputPtr *C.double,
) C.int {
	setLastError(nil)
	booster, err := fetchBooster(uint64(handle))
	if err != nil {
		setLastError(err)
		return 1
	}

	featuresInter, err := buildMatrix(featuresInterPtr, rows, interCols)
	if err != nil {
		setLastError(err)
		return 2
	}

	var featuresExtra [][]float64
	if extraCols > 0 {
		featuresExtra, err = buildMatrix(featuresExtraPtr, rows, extraCols)
		if err != nil {
			setLastError(err)
			return 3
		}
	}

	preds, err := booster.Predict(featuresInter, featuresExtra)
	if err != nil {
		setLastError(err)
		return 4
	}

	out := unsafe.Slice((*float64)(unsafe.Pointer(outputPtr)), int(rows))
	copy(out, preds)
	return 0
}

//export GetLastError
func GetLastError() *C.char {
	errStr := getLastError()
	if errStr == "" {
		return nil
	}
	return C.CString(errStr)
}

//export FreeCString
func FreeCString(str *C.char) {
	if str != nil {
		C.free(unsafe.Pointer(str))
	}
}

func main() {}
