package gorgonia

import (
	"fmt"
	"hash"
	"math"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type yoloOp struct {
	mask       []int
	anchors    []int
	inpDim     int
	numClasses int
	target     *Node
}

func newYoloOp(n *Node, anchors, mask []int, imheight, numclasses int, ignoreTresh float32, target ...*Node) *yoloOp {

	op := &yoloOp{
		mask:       mask,
		inpDim:     imheight,
		numClasses: numclasses,
	}
	if len(target) == 0 {
		for _, m := range mask {
			op.anchors = append(op.anchors, anchors[2*m], anchors[2*m+1])
		}
		op.mask = nil
	} else {
		op.anchors = anchors
		op.mask = mask
	}
	return op
}

// YOLOv3 https://arxiv.org/abs/1804.02767
func YOLOv3(x *Node, anchors, mask []int, imheight, numclasses int, ignoreTresh float32, target ...*Node) (*Node, error) {
	op := newYoloOp(x, anchors, mask, imheight, numclasses, ignoreTresh, target...)
	retVal, err := ApplyOp(op, x)
	return retVal, err
}

func (op *yoloOp) Arity() int {
	return 1
}

func (op *yoloOp) ReturnsPtr() bool { return false }

func (op *yoloOp) CallsExtern() bool { return false }

func (op *yoloOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Yolo{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) Hashcode() uint32 { return simpleHash(op) }

func (op *yoloOp) String() string {
	return fmt.Sprintf("Yolo{}(anchors: (%v))", op.anchors)
}
func (op *yoloOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}
func (op *yoloOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t)
}
func (op *yoloOp) OverwritesInput() int { return -1 }

func (op *yoloOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	var in tensor.Tensor
	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	if in.Shape().Dims() != 4 {
		return nil, errors.Errorf("Expected input to have 4 dimensions")
	}
	return in, nil
}

func sigmoidSlice(v tensor.View) error {
	switch v.Dtype() {
	case Float32:
		_, err := v.Apply(_sigmoidf32, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't do _sigmoidf32")
		}
	case Float64:
		_, err := v.Apply(_sigmoidf64, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't do _sigmoidf64")
		}
	default:
		return fmt.Errorf("Unsupported numeric type for YOLO v3 sigmoid function. Please use float64 or float32")
	}
	return nil
}

func expSlice(v tensor.View) error {
	switch v.Dtype() {
	case Float32:
		_, err := v.Apply(math32.Exp, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't do exp32")
		}
	case Float64:
		_, err := v.Apply(math.Exp, tensor.UseUnsafe())
		if err != nil {
			return errors.Wrap(err, "Can't do exp64")
		}
	default:
		return fmt.Errorf("Unsupported numeric type for YOLO v3 for exp function. Please use float64 or float32")
	}
	return nil
}

func (op *yoloOp) Do(inputs ...Value) (retVal Value, err error) {

	input, err := op.checkInput(inputs...)
	if err != nil {
		return nil, errors.Wrap(err, "Can't check input for YOLO v3")
	}
	batch := input.Shape()[0]
	stride := int(op.inpDim / input.Shape()[2])
	grid := int(op.inpDim / stride)
	bboxAttrs := 5 + op.numClasses
	numAnchors := len(op.anchors) / 2

	err = input.Reshape(batch, bboxAttrs*numAnchors, grid*grid)
	if err != nil {
		return nil, errors.Wrap(err, "Can't make reshape grid^2 for YOLO v3")
	}

	err = input.T(0, 2, 1)
	if err != nil {
		return nil, errors.Wrap(err, "Can't safely transponse input for YOLO v3")
	}
	err = input.Transpose()
	if err != nil {
		return nil, errors.Wrap(err, "Can't transponse input for YOLO v3")
	}
	err = input.Reshape(batch, grid*grid*numAnchors, bboxAttrs)
	if err != nil {
		return nil, errors.Wrap(err, "Can't reshape bbox for YOLO v3")
	}

	inputNumericType := input.Dtype()

	anchors := []int{}
	for i := 0; i < grid*grid; i++ {
		anchors = append(anchors, op.anchors...)
	}
	anchorsTensor := tensor.New(tensor.Of(inputNumericType), tensor.WithShape(1, grid*grid*numAnchors, 2))
	for i := range anchors {
		switch inputNumericType {
		case Float32:
			anchorsTensor.Set(i, float32(anchors[i]))
			break
		case Float64:
			anchorsTensor.Set(i, float64(anchors[i]))
		default:
			// currenty no do not handle this case
			break
		}
	}

	switch inputNumericType {
	case Float32:
		_, err = tensor.Div(anchorsTensor, float32(stride), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Div(...) for float32")
		}
	case Float64:
		_, err = tensor.Div(anchorsTensor, float64(stride), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Div(...) for float64")
		}
	default:
		return nil, fmt.Errorf("Unsupported numeric type for YOLO v3 in tensor.Div(...) function. Please use float64 or float32")
	}

	// Activation of x, y, and objects via sigmoid function
	slXY, err := input.Slice(nil, nil, S(0, 2))
	err = sigmoidSlice(slXY)
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate XY")
	}
	slClasses, err := input.Slice(nil, nil, S(4, 5+op.numClasses))
	err = sigmoidSlice(slClasses)
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate classes")
	}

	step := grid * numAnchors
	for i := 0; i < grid; i++ {
		vy, err := input.Slice(nil, S(i*step, i*step+step), S(1))
		if err != nil {
			return nil, errors.Wrap(err, "Can't slice while doing steps for grid")
		}
		switch inputNumericType {
		case Float32:
			_, err = tensor.Add(vy, float32(i), tensor.UseUnsafe())
			if err != nil {
				return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
			}
		case Float64:
			_, err = tensor.Add(vy, float64(i), tensor.UseUnsafe())
			if err != nil {
				return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float64; (1)")
			}
		default:
			return nil, fmt.Errorf("Unsupported numeric type for YOLO v3 for tensor.Add(...) function (1). Please use float64 or float32")
		}

		for n := 0; n < numAnchors; n++ {
			anchorsSlice, err := input.Slice(nil, S(i*numAnchors+n, input.Shape()[1], step), S(0))
			if err != nil {
				return nil, errors.Wrap(err, "Can't slice anchors while doing steps for grid")
			}
			switch inputNumericType {
			case Float32:
				_, err = tensor.Add(anchorsSlice, float32(i), tensor.UseUnsafe())
				if err != nil {
					return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float32; (1)")
				}
			case Float64:
				_, err = tensor.Add(anchorsSlice, float64(i), tensor.UseUnsafe())
				if err != nil {
					return nil, errors.Wrap(err, "Can't do tensor.Add(...) for float64; (2)")
				}
			default:
				return nil, fmt.Errorf("Unsupported numeric type for YOLO v3 in tensor.Add(...) function (2). Please use float64 or float32")
			}
		}
	}

	vhw, err := input.Slice(nil, nil, S(2, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(2,4)")
	}

	err = expSlice(vhw)
	if err != nil {
		return nil, errors.Wrap(err, "Can't activate grid")
	}

	_, err = tensor.Mul(vhw, anchorsTensor, tensor.UseUnsafe())
	if err != nil {
		return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for anchors")
	}

	vv, err := input.Slice(nil, nil, S(0, 4))
	if err != nil {
		return nil, errors.Wrap(err, "Can't do slice on input S(0,4)")
	}

	switch inputNumericType {
	case Float32:
		_, err = tensor.Mul(vv, float32(stride), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for float32")
		}
	case Float64:
		_, err = tensor.Mul(vv, float64(stride), tensor.UseUnsafe())
		if err != nil {
			return nil, errors.Wrap(err, "Can't do tensor.Mul(...) for float64")
		}
	default:
		return nil, fmt.Errorf("Unsupported numeric type for YOLO v3 in tensor.Mul(...) function. Please use float64 or float32")
	}

	return input, nil
}
