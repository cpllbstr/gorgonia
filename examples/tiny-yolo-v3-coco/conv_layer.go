package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type convLayer struct {
	filters        int
	padding        int
	kernelSize     int
	stride         int
	activation     string
	batchNormalize int
	bias           bool

	layerIndex int

	//learnables
	means, vars *gorgonia.Node

	//loadables
	kernels, gamma, beta, biases *gorgonia.Node

	convOut, bnOut, actOut *gorgonia.Node

	outShape tensor.Shape

	bnOp *gorgonia.BatchNormOp
}

func (l *convLayer) String() string {
	return fmt.Sprintf(
		"Convolution layer: Filters->%[1]d Padding->%[2]d Kernel->%[3]dx%[3]d Stride->%[4]d Activation->%[5]s Batch->%[6]d Bias->%[7]t",
		l.filters, l.padding, l.kernelSize, l.stride, l.activation, l.batchNormalize, l.bias,
	)
}

func (l *convLayer) Type() string {
	return "convolutional"
}

func (l *convLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {
	l.kernels = gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(l.filters, input[0].Shape()[1], l.kernelSize, l.kernelSize), gorgonia.WithName(fmt.Sprintf("conv_%d", l.layerIndex)))

	var err error
	l.convOut, err = gorgonia.Conv2d(input[0], l.kernels, tensor.Shape{l.kernelSize, l.kernelSize}, []int{l.padding, l.padding}, []int{l.stride, l.stride}, []int{1, 1})
	if err != nil {
		return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare convolution operation")
	}

	if l.bias {
		l.biases = gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(l.convOut.Shape()...), gorgonia.WithName(fmt.Sprintf("bias_%d", l.layerIndex)))
	}

	if l.batchNormalize > 0 {
		l.beta = gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(l.convOut.Shape().Clone()...), gorgonia.WithName(fmt.Sprintf("beta_%d", l.layerIndex)))
		l.gamma = gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(l.convOut.Shape().Clone()...), gorgonia.WithName(fmt.Sprintf("gamma_%d", l.layerIndex)))
		l.vars = gorgonia.NewTensor(g, tensor.Float32, 1, gorgonia.WithShape(l.filters), gorgonia.WithName(fmt.Sprintf("vars_%d", l.layerIndex)))
		l.means = gorgonia.NewTensor(g, tensor.Float32, 1, gorgonia.WithShape(l.filters), gorgonia.WithName(fmt.Sprintf("means_%d", l.layerIndex)))
		l.bnOut, l.gamma, l.beta, l.bnOp, err = gorgonia.BatchNorm(l.convOut, l.gamma, l.beta, 0.1, 10e-5)
		if err != nil {
			return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare batch normalization operation")
		}
		l.outShape = l.bnOut.Shape()
	} else {
		fmt.Println("SHP:", l.biases.Shape(), l.convOut.Shape())
		l.convOut, err = gorgonia.Add(l.convOut, l.biases)
		if err != nil {
			panic(err)
		}
		l.outShape = l.convOut.Shape()
		l.bnOut = l.convOut
	}

	if l.activation == "leaky" {
		var err error
		l.actOut, err = gorgonia.LeakyRelu(l.bnOut, 0.1)
		if err != nil {
			return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare activation operation")
		}
		return l.actOut, nil
	}
	return l.convOut, nil
}
