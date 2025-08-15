// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package neckercube

import (
	"github.com/emer/leabra/v2/leabra"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = leabra.LayerSheets{
	"Base": { // TODO: Do we *not* need Path: No learning with "Path.Learn.Learn: False" in here
		{Sel: "Layer", Doc: "generic params for all layers",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.ActAvg.Init = 0.35
				ly.Inhib.ActAvg.Fixed = true
				ly.Inhib.Layer.Gi = 1.4
				ly.Inhib.Layer.FB = 0.5
				// ly.Inhib.Layer.FBTau = 3 // this is key for smoothing bumps // Same or diff from .FB? ~ Q TODO
				ly.Act.Clamp.Hard = false
				ly.Act.Clamp.Gain = 0.1
				// ly.Act.XX1.Gain = 40 // more graded -- key
				ly.Act.Dt.VmTau = 6 // a bit slower -- not as effective as FBTau
				// ly.Act.Noise.Dist = Gaussian // Undefed for now TODO Q
				ly.Act.Noise.Var = 0.01
				// ly.Act.Noise.Type = GeNoise // Undefd for now TODO Q
				ly.Act.Noise.Fixed =  false
				ly.Act.KNa.Slow.Rise = 0.005
				ly.Act.KNa.Slow.Max = 0.2
				ly.Act.Gbar.K = 1.2
				ly.Act.Gbar.L = 0.1
			}},
		// {Sel: ".Id", Doc: "specific inhibition for identity, name",
		// 	Set: func(ly *leabra.LayerParams) {
		// 		ly.Inhib.Layer.Gi = 4.0
		// }},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = leabra.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "basic path params",
			Set: func(pt *leabra.PathParams) {
				pt.Learn.Learn = false
			}},
	},
}
