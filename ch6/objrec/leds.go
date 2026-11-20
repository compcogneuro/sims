// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"
	"image/color"

	"cogentcore.org/core/colors"
	"cogentcore.org/core/paint"
)

// LEDraw renders old-school "LED" style "letters" composed of a set of horizontal
// and vertical elements.  All possible such combinations of 3 out of 6 line segments are created.
// Renders using SVG.
type LEDraw struct { //types:add

	// line width of LEDraw as percent of display size
	Width float32 `default:"4"`

	// size of overall LED as proportion of overall image size
	Size float32 `default:"0.6"`

	// color name for drawing lines
	LineColor color.RGBA

	// color name for background
	BgColor color.RGBA

	// size of image to render
	ImgSize image.Point

	// rendered image
	Image *image.RGBA `display:"-"`

	// painting context object
	Paint *paint.Context `display:"-"`
}

func (ld *LEDraw) Defaults() {
	ld.ImgSize = image.Point{120, 120}
	ld.Width = 4
	ld.Size = 0.6
	ld.LineColor = colors.White
	ld.BgColor = colors.Black
}

// Init ensures that the image is created and of the right size, and renderer is initialized
func (ld *LEDraw) Init() {
	if ld.ImgSize.X == 0 || ld.ImgSize.Y == 0 {
		ld.Defaults()
	}
	if ld.Image != nil {
		cs := ld.Image.Bounds().Size()
		if cs != ld.ImgSize {
			ld.Image = nil
		}
	}
	if ld.Image == nil {
		ld.Image = image.NewRGBA(image.Rectangle{Max: ld.ImgSize})
	}
	ld.Paint = paint.NewContextFromImage(ld.Image)
	ld.Paint.StrokeStyle.Width.Pw(ld.Width)
	ld.Paint.StrokeStyle.Color = colors.Uniform(ld.LineColor)
	ld.Paint.FillStyle.Color = colors.Uniform(ld.BgColor)
	ld.Paint.SetUnitContextExt(ld.ImgSize)
}

// Clear clears the image with BgColor
func (ld *LEDraw) Clear() {
	if ld.Image == nil {
		ld.Init()
	}
	ld.Paint.Clear()
}

// DrawSeg draws one segment
func (ld *LEDraw) DrawSeg(seg LEDSegs) {
	ctrX := float32(ld.ImgSize.X) * 0.5
	ctrY := float32(ld.ImgSize.Y) * 0.5
	szX := ctrX * ld.Size
	szY := ctrY * ld.Size
	// note: top-zero coordinates
	switch seg {
	case Bottom:
		ld.Paint.DrawLine(ctrX-szX, ctrY+szY, ctrX+szX, ctrY+szY)
	case Left:
		ld.Paint.DrawLine(ctrX-szX, ctrY-szY, ctrX-szX, ctrY+szY)
	case Right:
		ld.Paint.DrawLine(ctrX+szX, ctrY-szY, ctrX+szX, ctrY+szY)
	case Top:
		ld.Paint.DrawLine(ctrX-szX, ctrY-szY, ctrX+szX, ctrY-szY)
	case CenterH:
		ld.Paint.DrawLine(ctrX-szX, ctrY, ctrX+szX, ctrY)
	case CenterV:
		ld.Paint.DrawLine(ctrX, ctrY-szY, ctrX, ctrY+szY)
	}
	ld.Paint.Stroke()
}

// DrawLED draws one LED of given number, based on LEDdata
func (ld *LEDraw) DrawLED(num int) {
	led := LEData[num]
	for _, seg := range led {
		if seg == NoSeg {
			continue
		}
		ld.DrawSeg(seg)
	}
}

//////////////////////////////////////////////////////////////////////////
//  LED data

// LEDSegs are the led segments
type LEDSegs int32

const (
	Bottom LEDSegs = iota
	Left
	Right
	Top
	CenterH
	CenterV
	LEDSegsN
	NoSeg LEDSegs = -1 // sentinel for unused slots in 4-element segment arrays
)

// LEData contains the segment combinations for each LED object.
// The first 20 entries are the original 3-segment combinations (all C(6,3)=20 unique patterns).
// Entries 20-24 are new 4-segment combinations that form more complex stimuli.
// Entries 23 and 24 are reserved as "novel" objects not trained in the first phase.
var LEData = [][4]LEDSegs{
	{CenterH, CenterV, Right, NoSeg},
	{Top, CenterV, Bottom, NoSeg},
	{Top, Right, Bottom, NoSeg},
	{Bottom, CenterV, Right, NoSeg},
	{Left, CenterH, Right, NoSeg},

	{Left, CenterV, CenterH, NoSeg},
	{Left, CenterV, Right, NoSeg},
	{Left, CenterV, Bottom, NoSeg},
	{Left, CenterH, Top, NoSeg},
	{Left, CenterH, Bottom, NoSeg},

	{Top, CenterV, Right, NoSeg},
	{Bottom, CenterV, CenterH, NoSeg},
	{Right, CenterH, Bottom, NoSeg},
	{Top, CenterH, Bottom, NoSeg},
	{Left, Top, Right, NoSeg},

	{Top, CenterH, Right, NoSeg},
	{Left, CenterV, Top, NoSeg},
	{Top, Left, Bottom, NoSeg},
	{Left, Bottom, Right, NoSeg},
	{Top, CenterV, CenterH, NoSeg},

	// New 4-segment patterns (indices 20-24)
	{Bottom, Left, Right, Top},   // 20: rectangle (box outline)
	{Left, Right, CenterH, CenterV}, // 21: H crossed with I
	{Bottom, Top, CenterH, CenterV}, // 22: double crossbar (like = with verticals)
	{Bottom, Right, CenterH, CenterV}, // 23: novel -- complex bracket right
	{Left, Top, CenterH, CenterV},    // 24: novel -- complex bracket left
}
