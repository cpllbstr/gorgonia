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
type YoloTrainer struct {
	op *yoloOp
}

// ActivateTrainingMode Activates training mode for yoloOP
func (yt *YoloTrainer) ActivateTrainingMode() {
	yt.op.trainMode = true
}

// DisableTrainingMode Disables training mode for yoloOP
func (yt *YoloTrainer) DisableTrainingMode() {
	yt.op.trainMode = false
}

// SetTarget sets []float32 as desired target for yoloOP
func (yt *YoloTrainer) SetTarget(target []float32) {
	preparedNumOfElements := yt.op.gridSize * yt.op.gridSize * len(yt.op.masks) * (5 + yt.op.numClasses)
	if yt.op.training == nil {
		fmt.Println("Training parameters were not set. Initializing empty slices....")
		yt.op.training = &yoloTraining{}
	}
	yt.op.training.scales = make([]float32, preparedNumOfElements)
	yt.op.training.targets = make([]float32, preparedNumOfElements)
	for i := range yt.op.training.scales {
		yt.op.training.scales[i] = 1
	}

	gridSizeF32 := float32(yt.op.gridSize)
	yt.op.bestAnchors = getBestAnchors_f32(target, yt.op.anchors, yt.op.masks, yt.op.dimensions, gridSizeF32)
	for i := 0; i < len(yt.op.bestAnchors); i++ {
		scale := (2 - target[i*5+3]*target[i*5+4])
		giInt := yt.op.bestAnchors[i][1]
		gjInt := yt.op.bestAnchors[i][2]
		gx := invsigm32(target[i*5+1]*gridSizeF32 - float32(giInt))
		gy := invsigm32(target[i*5+2]*gridSizeF32 - float32(gjInt))
		gw := math32.Log(target[i*5+3]/yt.op.anchors[yt.op.bestAnchors[i][0]] + 1e-16)
		gh := math32.Log(target[i*5+4]/yt.op.anchors[yt.op.bestAnchors[i][0]+1] + 1e-16)
		bboxIdx := gjInt*yt.op.gridSize*(5+yt.op.numClasses)*len(yt.op.masks) + giInt*(5+yt.op.numClasses)*len(yt.op.masks) + yt.op.bestAnchors[i][0]*(5+yt.op.numClasses)
		yt.op.training.scales[bboxIdx] = scale
		yt.op.training.targets[bboxIdx] = gx
		yt.op.training.scales[bboxIdx+1] = scale
		yt.op.training.targets[bboxIdx+1] = gy
		yt.op.training.scales[bboxIdx+2] = scale
		yt.op.training.targets[bboxIdx+2] = gw
		yt.op.training.scales[bboxIdx+3] = scale
		yt.op.training.targets[bboxIdx+3] = gh
		yt.op.training.targets[bboxIdx+4] = 1
		for j := 0; j < yt.op.numClasses; j++ {
			if j == int(target[i*5]) {
				yt.op.training.targets[bboxIdx+5+j] = 1
			}
		}
	}
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
	inputs  []float32
	bboxes  []float32
	scales  []float32
	targets []float32
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
	return ret, YoloTrainer{op}, err
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
		op.training = &yoloTraining{}
	}
	op.training.inputs, err = convertTensorToFloat32(inputTensor)
	if err != nil {
		return nil, errors.Wrap(err, "Can't cast tensor to []float32 for inputs [Training mode]")
	}
	op.training.bboxes, err = convertTensorToFloat32(yoloBBoxes)
	if err != nil {
		return nil, errors.Wrap(err, "Can't cast tensor to []float32 for bboxes [Training mode]")
	}

	preparedYOLOout := prepareOutputYOLO_f32(
		op.training.inputs, op.training.bboxes,
		op.training.targets, op.training.scales,
		op.bestAnchors, op.masks,
		op.numClasses, op.dimensions, op.gridSize, op.ignoreTresh,
	)

	yoloTensor := &tensor.Dense{}
	switch inputNumericType {
	case Float32:
		yoloTensor = tensor.New(tensor.WithShape(1, op.gridSize*op.gridSize*len(op.masks), 5+op.numClasses), tensor.Of(tensor.Float32), tensor.WithBacking(preparedYOLOout))
	case Float64:
		return nil, fmt.Errorf("Float64 not handled yet for YOLO tensor [Training mode]")
	default:
		return nil, fmt.Errorf("yoloOp supports only Float32/Float64 types for YOLO tensor [Training mode]")
	}

	return yoloTensor, nil
}

func convertTensorToFloat32(in tensor.Tensor) (input32 []float32, err error) {
	input32 = make([]float32, 0)
	in.Reshape(in.Shape().TotalSize())
	for i := 0; i < in.Shape()[0]; i++ {
		var buf interface{}
		buf, err = in.At(i)
		switch in.Dtype() {
		case Float32:
			input32 = append(input32, buf.(float32))
			break
		case Float64:
			input32 = append(input32, float32(buf.(float64)))
			break
		default:
			return nil, fmt.Errorf("convertTensorToFloat32() supports only Float32/Float64 types of tensor")
		}
	}
	return input32, nil
}

func (op *yoloOp) evaluateYOLO_f32(input tensor.Tensor, batchSize, stride, grid, bboxAttrs, numAnchors int, currentAnchors []float32) (retVal tensor.Tensor, err error) {

	inputNumericType := input.Dtype()
	if inputNumericType != Float32 {
		return nil, fmt.Errorf("evaluateYOLO_f32() called with input tensor of type %v. Float32 is required", inputNumericType)
	}

	err = input.Reshape(batchSize, bboxAttrs*numAnchors, grid*grid)
	if err != nil {
		return nil, errors.Wrap(err, "Can't make reshape grid^2 for YOLO")
	}

	err = input.T(0, 2, 1)
	if err != nil {
		return nil, errors.Wrap(err, "Can't safely transponse input for YOLO")
	}
	err = input.Transpose()
	if err != nil {
		return nil, errors.Wrap(err, "Can't transponse input for YOLO")
	}
	err = input.Reshape(batchSize, grid*grid*numAnchors, bboxAttrs)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape bbox for YOLO")
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

func getBestIOU_f32(input, target []float32, numClasses, dims int) [][]float32 {
	ious := make([][]float32, 0)
	imgsize := float32(dims)
	for i := 0; i < len(input); i = i + numClasses + 5 {
		ious = append(ious, []float32{0, -1})
		r1 := rectifyBox_f32(input[i], input[i+1], input[i+2], input[i+3], dims)
		for j := 0; j < len(target); j = j + 5 {
			r2 := rectifyBox_f32(target[j+1]*imgsize, target[j+2]*imgsize, target[j+3]*imgsize, target[j+4]*imgsize, dims)
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

func prepareOutputYOLO_f32(input, yoloBoxes, target, scales []float32, bestAnchors [][]int, masks []int, numClasses, dims, gridSize int, ignoreTresh float32) []float32 {
	yoloBBoxes := make([]float32, len(yoloBoxes))
	bestIous := getBestIOU_f32(yoloBoxes, target, numClasses, dims)
	for i := 0; i < len(yoloBoxes); i = i + (5 + numClasses) {
		if bestIous[i/(5+numClasses)][0] <= ignoreTresh {
			yoloBBoxes[i+4] = bceLoss32(0, yoloBoxes[i+4])
		}
	}
	for i := 0; i < len(bestAnchors); i++ {
		if bestAnchors[i][0] != -1 {
			giInt := bestAnchors[i][1]
			gjInt := bestAnchors[i][2]
			boxi := gjInt*gridSize*(5+numClasses)*len(masks) + giInt*(5+numClasses)*len(masks) + bestAnchors[i][0]*(5+numClasses)
			yoloBBoxes[boxi] = mseLoss32(target[boxi], input[boxi], scales[boxi])
			yoloBBoxes[boxi+1] = mseLoss32(target[boxi+1], input[boxi+1], scales[boxi+1])
			yoloBBoxes[boxi+2] = mseLoss32(target[boxi+2], input[boxi+2], scales[boxi+2])
			yoloBBoxes[boxi+3] = mseLoss32(target[boxi+3], input[boxi+3], scales[boxi+3])
			for j := 0; j < numClasses+1; j++ {
				yoloBBoxes[boxi+4+j] = bceLoss32(target[boxi+4+j], yoloBoxes[boxi+4+j])
			}
		}
	}
	return yoloBBoxes
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

func mseLoss32(target, pred, scale float32) float32 {
	return math32.Pow(scale*(target-pred), 2) / 2.0
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
		op.f32(inGradData, outGradData, op.training.scales, op.training.inputs, op.training.targets, op.training.bboxes)
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

func (op *yoloDiffOp) f32(inGradData, outGradData, scales, inputs, targets, bboxes []float32) {
	for i := range inGradData {
		inGradData[i] = 0
	}
	for i := 0; i < len(outGradData); i = i + 5 + op.numClasses {
		for j := 0; j < 4; j++ {
			inGradData[i+j] = outGradData[i+j] * (scales[i+j] * scales[i+j] * (inputs[i+j] - targets[i+j]))
		}
		for j := 4; j < 5+op.numClasses; j++ {
			if outGradData[i+j] != 0 {
				if targets[i+j] == 0 {
					inGradData[i+j] = outGradData[i+j] * (bboxes[i+j])
				} else {
					inGradData[i+j] = outGradData[i+j] * (1 - bboxes[i+j])
				}
			}
		}
	}
}
func (op *yoloDiffOp) f64(inGradData, outGradData, scales, inputs, targets, bboxes []float64) {
	for i := range inGradData {
		inGradData[i] = 0
	}
	for i := 0; i < len(outGradData); i = i + 5 + op.numClasses {
		for j := 0; j < 4; j++ {
			inGradData[i+j] = outGradData[i+j] * (scales[i+j] * scales[i+j] * (inputs[i+j] - targets[i+j]))
		}
		for j := 4; j < 5+op.numClasses; j++ {
			if outGradData[i+j] != 0 {
				if targets[i+j] == 0 {
					inGradData[i+j] = outGradData[i+j] * (bboxes[i+j])
				} else {
					inGradData[i+j] = outGradData[i+j] * (1 - bboxes[i+j])
				}
			}
		}
	}
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
