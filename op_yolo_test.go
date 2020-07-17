package gorgonia

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestYolo(t *testing.T) {

	inputSize := 416
	numClasses := 80
	testAnchors := [][]int{
		{10, 13, 16, 30, 33, 23},
		{30, 51, 62, 45, 59, 119},
		{116, 90, 156, 198, 373, 326},
	}

	numpyInputs := []string{
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1input.[(10, 13), (16, 30), (33, 23)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1input.[(30, 61), (62, 45), (59, 119)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1input.[(116, 90), (156, 198), (373, 326)].npy",
	}

	numpyExpectedOutputs := []string{
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1output.[(10, 13), (16, 30), (33, 23)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1output.[(30, 61), (62, 45), (59, 119)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1output.[(116, 90), (156, 198), (373, 326)].npy",
	}

	for i := range testAnchors {
		// Read input values from numpy format
		input := tensor.New(tensor.Of(tensor.Float32))
		r, err := os.Open(numpyInputs[i])
		if err != nil {
			t.Error(err)
			return
		}
		err = input.ReadNpy(r)
		if err != nil {
			t.Error(err)
			return
		}

		// Read expected values from numpy format
		expected := tensor.New(tensor.Of(tensor.Float32))
		r, err = os.Open(numpyExpectedOutputs[i])
		if err != nil {
			t.Error(err)
			return
		}
		err = expected.ReadNpy(r)
		if err != nil {
			t.Error(err)
			return
		}

		// Load graph
		g := NewGraph()
		inputTensor := NewTensor(g, tensor.Float32, 4, WithShape(input.Shape()...), WithName("yolo"))
		// Prepare YOLOv3 node
		outNode, err := YOLOv3(inputTensor, testAnchors[i], inputSize, numClasses)
		if err != nil {
			t.Error(err)
			return
		}
		// Run operation
		vm := NewTapeMachine(g)
		if err := Let(inputTensor, input); err != nil {
			t.Error(err)
			return
		}
		vm.RunAll()
		vm.Close()

		fmt.Println(outNode.Value().Data(), "\n", expected.Data())

		// Check if everything is fine
		if !assert.Equal(t, outNode.Value().Data(), expected.Data(), "Output is not equal to expected value") {
			t.Error(fmt.Sprintf("Got: %v\nExpected: %v", outNode.Value(), expected))
		}
	}
}

/*
func TestYolo(t *testing.T) {

	inputSize := 416
	numClasses := 80
	testAnchors := [][]int{
		[]int{10, 13, 16, 30, 33, 23},
		[]int{30, 51, 62, 45, 59, 119},
		[]int{116, 90, 156, 198, 373, 326},
	}

	numpyInputs := []string{
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1input.[(10, 13), (16, 30), (33, 23)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1input.[(30, 61), (62, 45), (59, 119)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1input.[(116, 90), (156, 198), (373, 326)].npy",
	}

	numpyExpectedOutputs := []string{
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1output.[(10, 13), (16, 30), (33, 23)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1output.[(30, 61), (62, 45), (59, 119)].npy",
		"./examples/tiny-yolo-v3-coco/data/test_yolo_op/1output.[(116, 90), (156, 198), (373, 326)].npy",
	}

	for i := range testAnchors {
		// Read input values from numpy format
		input := tensor.New(tensor.Of(tensor.Float32))
		r, err := os.Open(numpyInputs[i])
		if err != nil {
			t.Error(err)
			return
		}
		err = input.ReadNpy(r)
		if err != nil {
			t.Error(err)
			return
		}

		// Read expected values from numpy format
		expected := tensor.New(tensor.Of(tensor.Float32))
		r, err = os.Open(numpyExpectedOutputs[i])
		if err != nil {
			t.Error(err)
			return
		}
		err = expected.ReadNpy(r)
		if err != nil {
			t.Error(err)
			return
		}

		// Load graph
		g := NewGraph()
		inputTensor := NewTensor(g, tensor.Float32, 4, WithShape(input.Shape()...), WithName("yolo"))
		// Prepare YOLOv3 node
		outNode, err := YOLOv3(inputTensor, testAnchors[i], inputSize, numClasses)
		if err != nil {
			t.Error(err)
			return
		}
		// Run operation
		vm := NewTapeMachine(g)
		if err := Let(inputTensor, input); err != nil {
			t.Error(err)
			return
		}
		vm.RunAll()
		vm.Close()

		fmt.Println(outNode.Value().Data(), "\n", expected.Data())

		// Check if everything is fine
		if !assert.Equal(t, outNode.Value().Data(), expected.Data(), "Output is not equal to expected value") {
			t.Error(fmt.Sprintf("Got: %v\nExpected: %v", outNode.Value(), expected))
		}
	}
}
*/
