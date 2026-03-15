// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/etensor/tensor"
	"github.com/emer/vision/v2/vfilter"
	"github.com/emer/vision/v2/vxform"
)

// LEDEnv generates images of old-school "LED" style "letters" composed of a set of horizontal
// and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
// Renders using SVG.
type LEDEnv struct {

	// name of this environment
	Name string

	// draws LEDs onto image
	Draw LEDraw

	// visual processing params
	Vis Vis

	// minimum LED number to draw (0-24)
	MinLED int `min:"0" max:"24"`

	// maximum LED number to draw (0-24)
	MaxLED int `min:"0" max:"24"`

	// current LED number that was drawn
	CurLED int `edit:"-"`

	// previous LED number that was drawn
	PrvLED int `edit:"-"`

	// random transform parameters
	XFormRand vxform.Rand

	// current -- prev transforms
	XForm vxform.XForm

	// trial is the step counter for items
	Trial env.Counter `display:"inline"`

	// original image prior to random transforms
	OrigImg tensor.Float32

	// CurLED one-hot output tensor
	Output tensor.Float32
}

func (ev *LEDEnv) Label() string { return ev.Name }

func (ev *LEDEnv) States() env.Elements {
	isz := ev.Draw.ImgSize
	sz := ev.Vis.V1AllTsr.Shape().Sizes
	nms := ev.Vis.V1AllTsr.Shape().Names
	els := env.Elements{
		{"Image", []int{isz.Y, isz.X}, []string{"Y", "X"}},
		{"V1", sz, nms},
		{"Output", []int{5, 5}, []string{"Y", "X"}},
	}
	return els
}

func (ev *LEDEnv) State(element string) tensor.Tensor {
	switch element {
	case "Image":
		vfilter.RGBToGrey(ev.Draw.Image, &ev.OrigImg, 0, false) // pad for filt, bot zero
		return &ev.OrigImg
	case "V1":
		return &ev.Vis.V1AllTsr
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *LEDEnv) Actions() env.Elements {
	return nil
}

func (ev *LEDEnv) Defaults() {
	ev.Draw.Defaults()
	ev.Vis.Defaults()
	ev.XFormRand.TransX.Set(-0.25, 0.25)
	ev.XFormRand.TransY.Set(-0.25, 0.25)
	ev.XFormRand.Scale.Set(0.7, 1)
	ev.XFormRand.Rot.Set(-3.6, 3.6)
}

func (ev *LEDEnv) Init(run int) {
	ev.Draw.Init()
	ev.Trial.Scale = etime.Trial
	ev.Trial.Init()
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Output.SetShape([]int{5, 5}, "Y", "X")
}

func (ev *LEDEnv) Step() bool {
	ev.Trial.Incr()
	ev.DrawRandLED()
	ev.FilterImg()
	// debug only:
	// vfilter.RGBToGrey(ev.Draw.Image, &ev.OrigImg, 0, false) // pad for filt, bot zero
	return true
}

// DoObject renders specific object (LED number)
func (ev *LEDEnv) DoObject(objno int) {
	ev.DrawLED(objno)
	ev.FilterImg()
}

func (ev *LEDEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*LEDEnv)(nil)

// String returns the string rep of the LED env state
func (ev *LEDEnv) String() string {
	return fmt.Sprintf("Obj: %02d, %s", ev.CurLED, ev.XForm.String())
}

// SetOutput sets the output LED bit
func (ev *LEDEnv) SetOutput(out int) {
	ev.Output.SetZeros()
	ev.Output.SetFloat1D(out, 1)
}

// OutErr scores the output activity of network, returning the index of
// item with max overall activity, and 1 if that is error, 0 if correct.
// also returns a top-two error: if 2nd most active output was correct.
func (ev *LEDEnv) OutErr(tsr *tensor.Float32, corLED int) (maxi int, err, err2 float64) {
	nc := ev.Output.Len()
	maxi = 0
	maxv := 0.0
	for i := 0; i < nc; i++ {
		val := tsr.Float1D(i)
		if val > maxv {
			maxi = i
			maxv = val
		}
	}
	err = 1.0
	if maxi == corLED {
		err = 0
	}
	maxv2 := 0.0
	maxi2 := 0
	for i := 0; i < nc; i++ {
		if i == maxi { // skip top
			continue
		}
		val := tsr.Float1D(i)
		if val > maxv2 {
			maxi2 = i
			maxv2 = val
		}
	}
	err2 = err
	if maxi2 == corLED {
		err2 = 0
	}
	return
}

// DrawRandLED picks a new random LED and draws it
func (ev *LEDEnv) DrawRandLED() {
	rng := 1 + ev.MaxLED - ev.MinLED
	led := ev.MinLED + rand.Intn(rng)
	ev.DrawLED(led)
}

// DrawLED draw specified LED
func (ev *LEDEnv) DrawLED(led int) {
	ev.Draw.Clear()
	ev.Draw.DrawLED(led)
	ev.PrvLED = ev.CurLED
	ev.CurLED = led
	ev.SetOutput(ev.CurLED)
}

// FilterImg filters the image from LED
func (ev *LEDEnv) FilterImg() {
	ev.XFormRand.Gen(&ev.XForm)
	img := ev.XForm.Image(ev.Draw.Image)
	ev.Vis.Filter(img)
}
