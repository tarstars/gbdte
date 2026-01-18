// SPDX-License-Identifier: Apache-2.0

package main

/*
#cgo CFLAGS: -I.
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"io"
	"log"
	"math"
	"sync"
	"unsafe"

	ebl "github.com/tarstars/extra_bridged_boosting/golang/extra_boost/ebl"
	"gonum.org/v1/gonum/mat"
)

var (
	handleMu   sync.Mutex
	nextHandle uint64 = 1
	boosters          = make(map[uint64]*ebl.EBooster)

	monitorMu       sync.Mutex
	pendingMonitors []ebl.EMatrix

	lastErrorMu sync.Mutex
	lastError   string

	logSilenceOnce sync.Once
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

func storeBooster(b *ebl.EBooster) uint64 {
	handleMu.Lock()
	defer handleMu.Unlock()
	handle := nextHandle
	boosters[handle] = b
	nextHandle++
	return handle
}

func fetchBooster(handle uint64) (*ebl.EBooster, error) {
	handleMu.Lock()
	defer handleMu.Unlock()
	booster, ok := boosters[handle]
	if !ok {
		return nil, errors.New("invalid booster handle")
	}
	return booster, nil
}

//export FreeModel
func FreeModel(handle C.ulonglong) {
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

func sliceFromPtr(ptr *C.double, length int) ([]float64, error) {
	if length < 0 {
		return nil, errors.New("negative length")
	}
	if length == 0 {
		return nil, nil
	}
	if ptr == nil {
		return nil, errors.New("null pointer for non-empty slice")
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(ptr)), length), nil
}

func buildDense(ptr *C.double, rows, cols C.int) (*mat.Dense, error) {
	r := int(rows)
	c := int(cols)
	if r < 0 || c < 0 {
		return nil, errors.New("invalid matrix dimensions")
	}
	if r == 0 || c == 0 {
		return mat.NewDense(r, c, nil), nil
	}
	data, err := copyFloatSlice(ptr, r*c)
	if err != nil {
		return nil, err
	}
	return mat.NewDense(r, c, data), nil
}

func buildLoss(kind C.int) (ebl.SplitLoss, error) {
	switch kind {
	case 0:
		return ebl.MseLoss{}, nil
	case 1:
		return ebl.LogLoss{}, nil
	default:
		return nil, errors.New("unsupported loss kind")
	}
}

func makeRecordIDs(rows int) []int {
	ids := make([]int, rows)
	for i := range ids {
		ids[i] = i
	}
	return ids
}

//export RegisterLearningCurveDataset
func RegisterLearningCurveDataset(
	featuresInterPtr *C.double,
	rows C.int,
	interCols C.int,
	featuresExtraPtr *C.double,
	extraCols C.int,
	targetPtr *C.double,
	desc *C.char,
) C.int {
	setLastError(nil)

	if rows <= 0 {
		setLastError(errors.New("monitor rows must be positive"))
		return 1
	}

	inter, err := buildDense(featuresInterPtr, rows, interCols)
	if err != nil {
		setLastError(err)
		return 2
	}

	extra, err := buildDense(featuresExtraPtr, rows, extraCols)
	if err != nil {
		setLastError(err)
		return 3
	}

	target, err := buildDense(targetPtr, rows, 1)
	if err != nil {
		setLastError(err)
		return 4
	}

	matrix := ebl.EMatrix{
		FeaturesInter: inter,
		FeaturesExtra: extra,
		Target:        target,
		RecordIds:     makeRecordIDs(int(rows)),
	}

	if desc != nil {
		description := C.GoString(desc)
		matrix.SetDescription(description)
	}

	monitorMu.Lock()
	defer monitorMu.Unlock()
	pendingMonitors = append(pendingMonitors, matrix)
	return 0
}

//export TrainModel
func TrainModel(
	featuresInterPtr *C.double,
	rows C.int,
	interCols C.int,
	featuresExtraPtr *C.double,
	extraCols C.int,
	targetPtr *C.double,
	nStages C.int,
	regLambda C.double,
	maxDepth C.int,
	learningRate C.double,
	lossKind C.int,
	threadsNum C.int,
	unbalancedLoss C.double,
) C.ulonglong {
	setLastError(nil)
	logSilenceOnce.Do(func() {
		log.SetOutput(io.Discard)
	})

	if rows <= 0 {
		setLastError(errors.New("rows must be positive"))
		return 0
	}

	inter, err := buildDense(featuresInterPtr, rows, interCols)
	if err != nil {
		setLastError(err)
		return 0
	}

	extra, err := buildDense(featuresExtraPtr, rows, extraCols)
	if err != nil {
		setLastError(err)
		return 0
	}

	target, err := buildDense(targetPtr, rows, 1)
	if err != nil {
		setLastError(err)
		return 0
	}

	loss, err := buildLoss(lossKind)
	if err != nil {
		setLastError(err)
		return 0
	}

	params := ebl.EBoosterParams{
		Matrix: ebl.EMatrix{
			FeaturesInter: inter,
			FeaturesExtra: extra,
			Target:        target,
			RecordIds:     makeRecordIDs(int(rows)),
		},
		NStages:        int(nStages),
		RegLambda:      float64(regLambda),
		MaxDepth:       int(maxDepth),
		LearningRate:   float64(learningRate),
		LossKind:       loss,
		PrintMessages:  nil,
		ThreadsNum:     int(math.Max(1, float64(threadsNum))),
		UnbalancedLoss: float64(unbalancedLoss),
		Bias:           nil,
	}

	monitorMu.Lock()
	if len(pendingMonitors) > 0 {
		params.PrintMessages = append([]ebl.EMatrix(nil), pendingMonitors...)
		pendingMonitors = nil
	}
	monitorMu.Unlock()

	booster := ebl.NewEBooster(params)
	handle := storeBooster(booster)
	return C.ulonglong(handle)
}

func denseFromData(ptr *C.double, rows, cols C.int) (*mat.Dense, error) {
	return buildDense(ptr, rows, cols)
}

//export Predict
func Predict(
	handle C.ulonglong,
	featuresInterPtr *C.double,
	rows C.int,
	interCols C.int,
	featuresExtraPtr *C.double,
	extraCols C.int,
	outputPtr *C.double,
	treeLimit C.int,
) C.int {
	setLastError(nil)
	booster, err := fetchBooster(uint64(handle))
	if err != nil {
		setLastError(err)
		return 1
	}

	inter, err := denseFromData(featuresInterPtr, rows, interCols)
	if err != nil {
		setLastError(err)
		return 2
	}

	extra, err := denseFromData(featuresExtraPtr, rows, extraCols)
	if err != nil {
		setLastError(err)
		return 3
	}

	var limit *int
	if treeLimit > 0 {
		l := int(treeLimit)
		limit = &l
	}

	prediction := booster.PredictValue(inter, extra, limit)
	if prediction == nil {
		setLastError(errors.New("prediction failed"))
		return 4
	}

	outSlice, err := sliceFromPtr(outputPtr, int(rows))
	if err != nil {
		setLastError(err)
		return 5
	}
	copy(outSlice, prediction.RawMatrix().Data)
	return 0
}

//export SaveModel
func SaveModel(handle C.ulonglong, path *C.char) C.int {
	setLastError(nil)
	booster, err := fetchBooster(uint64(handle))
	if err != nil {
		setLastError(err)
		return 1
	}
	goPath := C.GoString(path)
	booster.Save(goPath)
	return 0
}

//export RenderTrees
func RenderTrees(handle C.ulonglong, prefix, figureType, directory *C.char) C.int {
	setLastError(nil)
	booster, err := fetchBooster(uint64(handle))
	if err != nil {
		setLastError(err)
		return 1
	}
	goPrefix := C.GoString(prefix)
	goFigureType := C.GoString(figureType)
	goDir := C.GoString(directory)
	if goPrefix == "" {
		goPrefix = "tree"
	}
	if goFigureType == "" {
		goFigureType = "svg"
	}
	if goDir == "" {
		goDir = "."
	}
	booster.RenderTrees(goPrefix, goFigureType, goDir)
	return 0
}

//export LoadModel
func LoadModel(path *C.char) C.ulonglong {
	setLastError(nil)
	goPath := C.GoString(path)
	booster := ebl.LoadModel(goPath)
	handle := storeBooster(&booster)
	return C.ulonglong(handle)
}

//export DumpLearningCurves
func DumpLearningCurves(handle C.ulonglong, path *C.char) C.int {
	setLastError(nil)
	booster, err := fetchBooster(uint64(handle))
	if err != nil {
		setLastError(err)
		return 1
	}
	goPath := C.GoString(path)
	booster.DumpLearningCurves(goPath)
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
