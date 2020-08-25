package gorgonia

import (
	"fmt"
	"hash"
	"image"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// YoloTrainer Wrapper around yoloOP
// It has method for setting desired bboxes as output of network
type YoloTrainer interface {
	ActivateTrainingMode()
	DisableTrainingMode()
	SetTarget([]float32)
}

// ActivateTrainingMode Activates training mode for yoloOP
func (op *yoloOp) ActivateTrainingMode() {
	op.trainMode = true
}

// DisableTrainingMode Disables training mode for yoloOP
func (op *yoloOp) DisableTrainingMode() {
	op.trainMode = false
}

// SetTarget sets []float32 as desired target for yoloOP
func (op *yoloOp) SetTarget(target []float32) {
	preparedNumOfElements := op.gridSize * op.gridSize * len(op.masks) * (5 + op.numClasses)
	if op.training == nil {
		fmt.Println("Training parameters were not set. Initializing empty slices....")
		op.training = &yoloTraining{}
	}

	tmpScales := make([]float32, preparedNumOfElements)
	for i := range tmpScales {
		tmpScales[i] = 1
	}
	tmpTargets := make([]float32, preparedNumOfElements)

	gridSizeF32 := float32(op.gridSize)
	op.bestAnchors = getBestAnchors_f32(target, op.anchors, op.masks, op.dimensions, gridSizeF32)
	for i := 0; i < len(op.bestAnchors); i++ {
		scale := (2 - target[i*5+3]*target[i*5+4])
		giInt := op.bestAnchors[i][1]
		gjInt := op.bestAnchors[i][2]
		gx := invsigm32(target[i*5+1]*gridSizeF32 - float32(giInt))
		gy := invsigm32(target[i*5+2]*gridSizeF32 - float32(gjInt))
		bestAnchor := op.masks[op.bestAnchors[i][0]] * 2
		gw := math32.Log(target[i*5+3]/op.anchors[bestAnchor] + 1e-16)
		gh := math32.Log(target[i*5+4]/op.anchors[bestAnchor+1] + 1e-16)
		bboxIdx := gjInt*op.gridSize*(5+op.numClasses)*len(op.masks) + giInt*(5+op.numClasses)*len(op.masks) + op.bestAnchors[i][0]*(5+op.numClasses)
		tmpScales[bboxIdx] = scale
		tmpTargets[bboxIdx] = gx
		tmpScales[bboxIdx+1] = scale
		tmpTargets[bboxIdx+1] = gy
		tmpScales[bboxIdx+2] = scale
		tmpTargets[bboxIdx+2] = gw
		tmpScales[bboxIdx+3] = scale
		tmpTargets[bboxIdx+3] = gh
		tmpTargets[bboxIdx+4] = 1.0
		for j := 0; j < op.numClasses; j++ {
			if j == int(target[i*5]) {
				tmpTargets[bboxIdx+5+j] = 1.0
			}
		}
	}

	op.training.scales = tensor.New(tensor.WithShape(preparedNumOfElements), tensor.Of(tensor.Float32), tensor.WithBacking(tmpScales))
	op.training.targets = tensor.New(tensor.WithShape(preparedNumOfElements), tensor.Of(tensor.Float32), tensor.WithBacking(tmpTargets))
}

type yoloOp struct {
	anchors     []float32
	masks       []int
	ignoreTresh float32
	dimensions  int
	numClasses  int
	trainMode   bool
	gridSize    int

	bestAnchors [][]int
	training    *yoloTraining
}

type yoloTraining struct {
	inputs  tensor.Tensor
	bboxes  tensor.Tensor
	scales  tensor.Tensor
	targets tensor.Tensor
}

func newYoloOp(anchors []float32, masks []int, netSize, gridSize, numClasses int, ignoreTresh float32) *yoloOp {
	yoloOp := &yoloOp{
		anchors:     anchors,
		dimensions:  netSize,
		gridSize:    gridSize,
		numClasses:  numClasses,
		ignoreTresh: ignoreTresh,
		masks:       masks,
		trainMode:   false,
		training:    &yoloTraining{},
	}
	return yoloOp
}

// YOLOv3 https://arxiv.org/abs/1804.02767
func YOLOv3(input *Node, anchors []float32, masks []int, netSize, numClasses int, ignoreTresh float32, targets ...*Node) (*Node, YoloTrainer, error) {
	op := newYoloOp(anchors, masks, netSize, input.Shape()[2], numClasses, ignoreTresh)
	ret, err := ApplyOp(op, input)
	return ret, op, err
}

func (op *yoloOp) Arity() int {
	return 1
}

func (op *yoloOp) ReturnsPtr() bool { return false }

func (op *yoloOp) CallsExtern() bool { return false }

func (op *yoloOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "YOLO{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) Hashcode() uint32 { return simpleHash(op) }

func (op *yoloOp) String() string {
	return fmt.Sprintf("YOLO{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	shp := inputs[0].(tensor.Shape)
	if len(shp) < 4 {
		return nil, fmt.Errorf("InferShape() for YOLO must contain 4 dimensions")
	}
	s := shp.Clone()
	if op.trainMode {
		return []int{s[0], s[2] * s[3] * len(op.masks), (s[1] - 1) / len(op.masks)}, nil
	}
	return []int{s[0], s[2] * s[3] * len(op.masks), s[1] / len(op.masks)}, nil
}

func (op *yoloOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	o := newTensorType(3, a)
	return hm.NewFnType(t, o)
}

func (op *yoloOp) OverwritesInput() int { return -1 }

func (op *yoloOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, errors.Wrap(err, "Can't check arity for YOLO operation")
	}
	var in tensor.Tensor
	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Can't check YOLO input: expected input has to be a tensor")
	}
	if in.Shape().Dims() != 4 {
		return nil, errors.Errorf("Can't check YOLO input: expected input must have 4 dimensions")
	}
	return in, nil
}

func (op *yoloOp) Do(inputs ...Value) (retVal Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, errors.Wrap(err, "Can't check YOLO input")
	}
	batchSize := inputTensor.Shape()[0]
	stride := op.dimensions / inputTensor.Shape()[2]
	gridSize := inputTensor.Shape()[2]
	bboxAttributes := 5 + op.numClasses
	numAnchors := len(op.anchors) / 2
	currentAnchors := []float32{}
	for i := range op.masks {
		if op.masks[i] >= numAnchors {
			return nil, fmt.Errorf("Incorrect mask %v for anchors in YOLO layer", op.masks)
		}
		currentAnchors = append(currentAnchors, op.anchors[i*2], op.anchors[i*2+1])
	}

	inputNumericType := inputTensor.Dtype()

	err = inputTensor.Reshape(batchSize, bboxAttributes*numAnchors, gridSize*gridSize)
	if err != nil {
		return nil, errors.Wrap(err, "Can't make reshape grid^2 for YOLO")
	}

	err = inputTensor.T(0, 2, 1)
	if err != nil {
		return nil, errors.Wrap(err, "Can't safely transponse input for YOLO")
	}
	err = inputTensor.Transpose()
	if err != nil {
		return nil, errors.Wrap(err, "Can't transponse input for YOLO")
	}
	err = inputTensor.Reshape(batchSize, gridSize*gridSize*numAnchors, bboxAttributes)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape bbox for YOLO")
	}

	// Just inference without backpropagation
	if !op.trainMode {
		switch inputNumericType {
		case Float32:
			return op.evaluateYOLO_f32(inputTensor, batchSize, stride, gridSize, bboxAttributes, len(op.masks), currentAnchors)
		case Float64:
			return nil, fmt.Errorf("Float64 not handled yet")
		default:
			return nil, fmt.Errorf("yoloOp supports only Float32/Float64 types")
		}
	}

	// Training mode
	inputTensorCopy := inputTensor.Clone().(tensor.Tensor)
	var yoloBBoxes tensor.Tensor
	switch inputNumericType {
	case Float32:
		yoloBBoxes, err = op.evaluateYOLO_f32(inputTensorCopy, batchSize, stride, gridSize, bboxAttributes, len(op.masks), currentAnchors)
		if err != nil {
			return nil, errors.Wrap(err, "Can't evaluate YOLO [Training mode]")
		}
	case Float64:
		return nil, fmt.Errorf("Float64 not handled yet [Training mode]")
	default:
		return nil, fmt.Errorf("yoloOp supports only Float32/Float64 types [Training mode]")
	}

	if op.training == nil {
		return nil, fmt.Errorf("Nil pointer on training params in yoloOp [Training mode]")
	}
	err = inputTensor.Reshape(inputTensor.Shape().TotalSize())
	if err != nil {
		return nil, errors.Wrap(err, "Can't cast tensor to []float32 for inputs [Training mode]")
	}
	op.training.inputs = inputTensor
	err = yoloBBoxes.Reshape(yoloBBoxes.Shape().TotalSize())
	if err != nil {
		return nil, errors.Wrap(err, "Can't cast tensor to []float32 for bboxes [Training mode]")
	}
	op.training.bboxes = yoloBBoxes

	return prepareOutputYOLOTensors(
		op.training.inputs, op.training.bboxes,
		op.training.targets, op.training.scales,
		op.bestAnchors, op.masks,
		op.numClasses, op.dimensions, op.gridSize, op.ignoreTresh,
	)
}

func (op *yoloOp) evaluateYOLO_f32(input tensor.Tensor, batchSize, stride, grid, bboxAttrs, numAnchors int, currentAnchors []float32) (retVal tensor.Tensor, err error) {

	inputNumericType := input.Dtype()
	if inputNumericType != Float32 {
		return nil, fmt.Errorf("evaluateYOLO_f32() called with input tensor of type %v. Float32 is required", inputNumericType)
	}

	// Activation of x, y, and objects via sigmoid function
	slXY, err := input.Slice(nil, nil, S(0, 2))
	_, err = slXY.Apply(_sigmoidf32, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate XY due _sigmoidf32 error")
	}

	slClasses, err := input.Slice(nil, nil, S(4, 5+op.numClasses))

	_, err = slClasses.Apply(_sigmoidf32, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate classes due _sigmoidf32 error")
	}

	step := grid * numAnchors
	for i := 0; i < grid; i++ {

		vy, err := input.Slice(nil, S(i*step, i*step+step), S(1))
		if err != nil {
			return nil, errors.Wrap(err, "Can't slice while doing steps for grid")
		}

		_, err = tensor.Add(vy, float32(i), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
		}

		for n := 0; n < numAnchors; n++ {
			anchorsSlice, err := input.Slice(nil, S(i*numAnchors+n, input.Shape()[1], step), S(0))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice anchors while doing steps for grid")
			}
			_, err = tensor.Add(anchorsSlice, float32(i), tensor.UseUnsafe())
			if err != nil {
				return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
			}
		}

	}

	anchors := []float32{}
	for i := 0; i < grid*grid; i++ {
		anchors = append(anchors, currentAnchors...)
	}

	anchorsTensor := tensor.New(tensor.Of(Float32), tensor.WithShape(1, grid*grid*numAnchors, 2))
	for i := range anchors {
		anchorsTensor.Set(i, anchors[i])
	}

	_, err = tensor.Div(anchorsTensor, float32(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Div(...) for float32")
	}

	vhw, err := input.Slice(nil, nil, S(2, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(2,4)")
	}

	_, err = vhw.Apply(math32.Exp, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't apply exp32 to YOLO operation")
	}

	_, err = tensor.Mul(vhw, anchorsTensor, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for anchors")
	}

	vv, err := input.Slice(nil, nil, S(0, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(0,4)")
	}

	_, err = tensor.Mul(vv, float32(stride), tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for float32")
	}

	return input, nil
}

func iou_f32(r1, r2 image.Rectangle) float32 {
	intersection := r1.Intersect(r2)
	interArea := intersection.Dx() * intersection.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	return float32(interArea) / float32(r1Area+r2Area-interArea)
}

func getBestIOU_f32(input, target tensor.Tensor, numClasses, dims int) [][]float32 {
	ious := make([][]float32, 0)
	imgsize := float32(dims)
	for i := 0; i < input.Shape().TotalSize(); i = i + numClasses + 5 {
		ious = append(ious, []float32{0, -1})
		input_0, _ := input.At(i)
		input_1, _ := input.At(i + 1)
		input_2, _ := input.At(i + 2)
		input_3, _ := input.At(i + 3)
		r1 := rectifyBox_f32(input_0.(float32), input_1.(float32), input_2.(float32), input_3.(float32), dims)
		for j := 0; j < target.Shape().TotalSize(); j = j + 5 {

			target_0, _ := target.At(j + 1)
			target_1, _ := target.At(j + 2)
			target_2, _ := target.At(j + 3)
			target_3, _ := target.At(i + 4)

			r2 := rectifyBox_f32(target_0.(float32)*imgsize, target_1.(float32)*imgsize, target_2.(float32)*imgsize, target_3.(float32)*imgsize, dims)
			curiou := iou_f32(r1, r2)
			if curiou > ious[i/(5+numClasses)][0] {
				ious[i/(5+numClasses)][0] = curiou
				ious[i/(5+numClasses)][1] = float32(j / 5)
			}
		}
	}
	return ious
}

func getBestAnchors_f32(target []float32, anchors []float32, masks []int, dims int, gridSize float32) [][]int {
	bestAnchors := make([][]int, len(target)/5)
	imgsize := float32(dims)
	for j := 0; j < len(target); j = j + 5 {
		targetRect := rectifyBox_f32(0, 0, target[j+3]*imgsize, target[j+4]*imgsize, dims) //not absolutely confident in rectangle sizes
		bestIOU := float32(0.0)
		bestAnchors[j/5] = make([]int, 3)
		for i := 0; i < len(anchors); i = i + 2 {
			anchorRect := rectifyBox_f32(0, 0, anchors[i], anchors[i+1], dims)
			currentIOU := iou_f32(anchorRect, targetRect)
			if currentIOU >= bestIOU {
				bestAnchors[j/5][0] = i
				bestIOU = currentIOU
			}
		}
		bestAnchors[j/5][0] = findIntElement(masks, bestAnchors[j/5][0]/2)
		if bestAnchors[j/5][0] != -1 {
			bestAnchors[j/5][1] = int(target[j+1] * gridSize)
			bestAnchors[j/5][2] = int(target[j+2] * gridSize)
		}
	}
	return bestAnchors
}

func prepareOutputYOLOTensors(input, yoloBoxes, target, scales tensor.Tensor, bestAnchors [][]int, masks []int, numClasses, dims, gridSize int, ignoreTresh float32) (tensor.Tensor, error) {
	tmpBBoxes := make([]float32, yoloBoxes.Shape().TotalSize())
	bestIous := getBestIOU_f32(yoloBoxes, target, numClasses, dims)
	for i := 0; i < yoloBoxes.Shape().TotalSize(); i = i + (5 + numClasses) {
		if bestIous[i/(5+numClasses)][0] <= ignoreTresh {
			bboxValue, err := yoloBoxes.At(i + 4)
			if err != nil {
				return nil, errors.Wrap(err, "Can't extract YOLO BBox value (before slicing)")
			}
			tmpBBoxes[i+4] = bceLoss32(0, bboxValue.(float32))
		}
	}
	for i := 0; i < len(bestAnchors); i++ {
		if bestAnchors[i][0] != -1 {
			giInt := bestAnchors[i][1]
			gjInt := bestAnchors[i][2]
			boxi := gjInt*gridSize*(5+numClasses)*len(masks) + giInt*(5+numClasses)*len(masks) + bestAnchors[i][0]*(5+numClasses)
			inputsSlice, err := input.Slice(S(boxi, boxi+4))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice inputs for BBoxes")
			}
			scalesSlice, err := scales.Slice(S(boxi, boxi+4))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice scales for BBoxes")
			}
			targetsSlice, err := target.Slice(S(boxi, boxi+4))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice targets for BBoxes")
			}
			mseTensor, err := mseLossTensors(targetsSlice, inputsSlice, scalesSlice)
			if err != nil {
				return nil, errors.Wrap(err, "Can't evaluate MSE for BBoxes")
			}
			mseIdx_0, err := mseTensor.At(0)
			if err != nil {
				return nil, errors.Wrap(err, "Can't extract MSE value at position 0 for BBoxes")
			}
			mseIdx_1, err := mseTensor.At(1)
			if err != nil {
				return nil, errors.Wrap(err, "Can't extract MSE value at position 1 for BBoxes")
			}
			mseIdx_2, err := mseTensor.At(2)
			if err != nil {
				return nil, errors.Wrap(err, "Can't extract MSE value at position 2 for BBoxes")
			}
			mseIdx_3, err := mseTensor.At(3)
			if err != nil {
				return nil, errors.Wrap(err, "Can't extract MSE value at position 3 for BBoxes")
			}

			tmpBBoxes[boxi] = mseIdx_0.(float32)
			tmpBBoxes[boxi+1] = mseIdx_1.(float32)
			tmpBBoxes[boxi+2] = mseIdx_2.(float32)
			tmpBBoxes[boxi+3] = mseIdx_3.(float32)

			for j := 0; j < numClasses+1; j++ {
				bboxValue, err := yoloBoxes.At(boxi + 4 + j)
				if err != nil {
					return nil, errors.Wrap(err, "Can't extract YOLO BBox value")
				}
				targetValue, err := target.At(boxi + 4 + j)
				if err != nil {
					return nil, errors.Wrap(err, "Can't extract target value")
				}
				tmpBBoxes[boxi+4+j] = bceLoss32(targetValue.(float32), bboxValue.(float32))
			}
		}
	}

	return tensor.New(tensor.WithShape(1, gridSize*gridSize*len(masks), 5+numClasses), tensor.Of(tensor.Float32), tensor.WithBacking(tmpBBoxes)), nil
}

func findIntElement(arr []int, ele int) int {
	for i := range arr {
		if arr[i] == ele {
			return i
		}
	}
	return -1
}

func rectifyBox_f32(x, y, h, w float32, imgSize int) image.Rectangle {
	return image.Rect(maxInt(int(x-w/2), 0), maxInt(int(y-h/2), 0), minInt(int(x+w/2+1), imgSize), minInt(int(y+h/2+1), imgSize))
}

func bceLoss32(target, pred float32) float32 {
	if target == 1.0 {
		return -(math32.Log(pred + 1e-16))
	}
	return -(math32.Log((1.0 - pred) + 1e-16))
}

func bceLossTensors(target float32, pred tensor.Tensor) (tensor.Tensor, error) {
	if target == 1.0 {
		return pred.Apply(_bceLossOverflow)
	}
	return pred.Apply(_bceLoss)
}

func _bceLossOverflow(x float32) float32 {
	return -(math32.Log(x + 1e-16))
}

func _bceLoss(x float32) float32 {
	return -(math32.Log((1.0 - x) + 1e-16))
}

func mseLossTensors(target, pred, scale tensor.Tensor) (tensor.Tensor, error) {
	sub, err := tensor.Sub(target, pred) // (target-pred)
	if err != nil {
		return nil, err
	}
	mul, err := tensor.Mul(scale, sub) // scale*(target-pred)
	if err != nil {
		return nil, err
	}
	pow2, err := tensor.Mul(mul, mul) // math32.Pow(scale*(target-pred), 2)
	if err != nil {
		return nil, err
	}
	div, err := pow2.Apply(_divBy2) // math32.Pow(scale*(target-pred), 2) / 2.0
	if err != nil {
		return nil, err
	}
	return div, err
}

func _divBy2(x float32) float32 {
	return float32(x / 2.0)
}

func invsigm32(target float32) float32 {
	return -math32.Log(1-target+1e-16) + math32.Log(target+1e-16)
}

type yoloDiffOp struct {
	yoloOp
}

func (op *yoloDiffOp) Arity() int { return 2 }
func (op *yoloDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	o := newTensorType(3, a)
	return hm.NewFnType(t, o, t)
}

func (op *yoloDiffOp) ReturnsPtr() bool     { return true }
func (op *yoloDiffOp) CallsExtern() bool    { return false }
func (op *yoloDiffOp) OverwritesInput() int { return -1 }
func (op *yoloDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}
func (op *yoloDiffOp) Do(inputs ...Value) (Value, error) {
	if op.training == nil {
		return nil, fmt.Errorf("Training parameters for yoloOp were not set")
	}
	if op.training.inputs == nil {
		return nil, fmt.Errorf("Training parameter 'inputs' for yoloOp were not set")
	}
	if op.training.scales == nil {
		return nil, fmt.Errorf("Training parameter 'scales' for yoloOp were not set")
	}
	if op.training.targets == nil {
		return nil, fmt.Errorf("Training parameter 'targets' for yoloOp were not set")
	}
	if op.training.bboxes == nil {
		return nil, fmt.Errorf("Training parameter 'bboxes' for yoloOp were not set")
	}

	in := inputs[0]
	output := inputs[1]
	inGrad := tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(output.Shape().Clone()...), tensor.WithEngine(in.(tensor.Tensor).Engine()))
	switch in.Dtype() {
	case tensor.Float32:
		inGradData := inGrad.Data().([]float32)
		outGradData := output.Data().([]float32)
		err := op.f32(inGradData, outGradData, op.training.scales, op.training.inputs, op.training.targets, op.training.bboxes)
		if err != nil {
			return nil, fmt.Errorf("yoloDiffOp can't evalute gradients")
		}
		break
	case tensor.Float64:
		return nil, fmt.Errorf("yoloDiffOp for Float64 is not implemented yet")
	default:
		return nil, fmt.Errorf("yoloDiffOp supports only Float32/Float64 types")
	}

	err := inGrad.Reshape(1, op.gridSize*op.gridSize, (op.numClasses+5)*len(op.masks))
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape in yoloDiffOp (1)")
	}
	err = inGrad.T(0, 2, 1)
	if err != nil {
		return nil, errors.Wrap(err, "Can't safely transponse in yoloDiffOp (1)")
	}
	err = inGrad.Transpose()
	if err != nil {
		return nil, errors.Wrap(err, "Can't transponse in yoloDiffOp (1)")
	}
	err = inGrad.Reshape(1, len(op.masks)*(5+op.numClasses), op.gridSize, op.gridSize)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape in yoloDiffOp (2)")
	}
	return inGrad, nil
}

func getScalar(inputs tensor.Tensor, index int) (tensor.Tensor, error) {
	inputValue, err := inputs.At(index)
	if err != nil {
		return nil, err
	}
	inT := inputValue.(tensor.Tensor)
	if !inT.IsScalar() {
		return nil, errors.Errorf("Error fetching data from tensor, data is not a Scalar")
	}
	return inT, nil
}

func (op *yoloDiffOp) f32(inGradData, outGradData []float32, scales, inputs, targets, bboxes tensor.Tensor) error {
	for i := range inGradData {
		inGradData[i] = 0
	}
	for i := 0; i < len(outGradData); i = i + 5 + op.numClasses {
		for j := 0; j < 4; j++ {
			inputValue, err := inputs.At(i + j)
			if err != nil {
				return errors.Wrap(err, "Can't extract input value")
			}
			scaleValue, err := scales.At(i + j)
			if err != nil {
				return errors.Wrap(err, "Can't extract scale value")
			}
			targetValue, err := targets.At(i + j)
			if err != nil {
				return errors.Wrap(err, "Can't extract target value")
			}
			inGradData[i+j] = outGradData[i+j] * (scaleValue.(float32) * scaleValue.(float32) * (inputValue.(float32) - targetValue.(float32)))
		}
		for j := 4; j < 5+op.numClasses; j++ {
			if outGradData[i+j] != 0 {
				inputValue, err := bboxes.At(i + j)
				if err != nil {
					return errors.Wrap(err, "Can't extract bbox value")
				}
				targetValue, err := targets.At(i + j)
				if err != nil {
					return errors.Wrap(err, "Can't extract target value")
				}
				if targetValue.(float32) == 0 {
					inGradData[i+j] = outGradData[i+j] * (inputValue.(float32))
				} else {
					inGradData[i+j] = outGradData[i+j] * (1 - inputValue.(float32))
				}
			}
		}
	}
	return nil
}

func (op *yoloDiffOp) Tf(inGrad, outGrad, scales, inputs, targets, bboxes tensor.Tensor) error {
	inGrad.Zero()
	one := tensor.Ones(inGrad.Dtype(), 1)
	for i := 0; i < outGrad.Size(); i += 5 + op.numClasses {
		for j := 0; j < 4; j++ {
			inputValue, err := getScalar(inputs, i+j)
			if err != nil {
				return errors.Wrap(err, "Can't extract input value")
			}
			scaleValue, err := getScalar(scales, i+j)
			if err != nil {
				return errors.Wrap(err, "Can't extract scale value")
			}
			targetValue, err := getScalar(targets, i+j)
			if err != nil {
				return errors.Wrap(err, "Can't extract target value")
			}
			outValue, err := getScalar(outGrad, i+j)
			if err != nil {
				return errors.Wrap(err, "Can't extract ourgrad value")
			}
			sub, err := tensor.Sub(inputValue, targetValue)
			if err != nil {
				return errors.Wrap(err, "Can't subtract")
			}
			sc2, err := tensor.Mul(scaleValue, scaleValue)
			if err != nil {
				return errors.Wrap(err, "Can't mul scale scale")
			}
			scsub, err := tensor.Mul(sc2, sub)
			if err != nil {
				return errors.Wrap(err, "Can't mul sc2 sub")
			}
			in, err := tensor.Mul(outValue, scsub)
			if err != nil {
				return errors.Wrap(err, "Can't mul out scsub")
			}
			inGrad.(*tensor.Dense).Set(i+j, in)
		}
		for j := 4; j < 5+op.numClasses; j++ {
			outValue, err := outGrad.At(i + j)
			if err != nil {
				return errors.Wrap(err, "Can't extract ourgrad value")
			}
			if outValue.(float32) != 0 {
				inputValue, err := getScalar(bboxes, i+j)
				if err != nil {
					return errors.Wrap(err, "Can't extract bbox value")
				}
				targetValue, err := targets.At(i + j)
				if err != nil {
					return errors.Wrap(err, "Can't extract target value")
				}
				outVal, err := getScalar(outGrad, i+j)
				if err != nil {
					return errors.Wrap(err, "Can't extract outGrad value")
				}
				if targetValue.(float32) == 0 {
					inGradVal, err := tensor.Mul(outVal, inputValue)
					if err != nil {
						return errors.Wrap(err, "Can't mul outval, inp value ")
					}
					inGrad.(*tensor.Dense).Set(i+j, inGradVal)
				} else {
					sub, err := tensor.Sub(one, inputValue)
					if err != nil {
						return errors.Wrap(err, "Can't sub one, inp value ")
					}
					inGradVal, err := tensor.Mul(outVal, sub)
					if err != nil {
						return errors.Wrap(err, "Can't mul outval, inp value ")
					}
					inGrad.(*tensor.Dense).Set(i+j, inGradVal)
				}
			}
		}
	}
	return nil
}

func (op *yoloOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	return fmt.Errorf("DoDiff for yoloOp is not implemented")
}

func (op *yoloOp) DiffWRT(inputs int) []bool { return []bool{true} }

func (op *yoloOp) SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	in := inputs[0]
	var op2 yoloOp
	op2 = *op
	diff := &yoloDiffOp{op2}

	var ret *Node
	if ret, err = ApplyOp(diff, in, grad); err != nil {
		return nil, err
	}
	return Nodes{ret}, nil
}
