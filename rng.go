package quantaspindle

import "math"

// ScenarioSeed derives scenario provenance using the versioned SplitMix64-v1
// finalizer: mix(seed XOR mix(id+0x9e3779b97f4a7c15)). CPU initializes its price
// stream from this value; other backends document their actual generator mapping.
func ScenarioSeed(seed, id uint64) uint64 { return mix(seed ^ mix(id+0x9e3779b97f4a7c15)) }

func mix(x uint64) uint64 {
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

type random struct {
	state    uint64
	spare    float64
	hasSpare bool
}

func (r *random) uniform() float64 {
	r.state += 0x9e3779b97f4a7c15
	// Midpoints of 2^52 bins keep both endpoints excluded in float64.
	return (float64(mix(r.state)>>12) + 0.5) * (1.0 / (1 << 52))
}
func (r *random) normal() float64 {
	if r.hasSpare {
		r.hasSpare = false
		return r.spare
	}
	u, v := r.uniform(), r.uniform()
	radius := math.Sqrt(-2 * math.Log(u))
	angle := 2 * math.Pi * v
	s, c := math.Sincos(angle)
	r.spare = radius * s
	r.hasSpare = true
	return radius * c
}
