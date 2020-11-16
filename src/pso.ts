import { Vector } from "./vector";
import * as nn from "./nn";

export interface Optimizable {
   cost(state: Vector): number;

   state(): Vector;
}

class Particle {
  pso: ParticleSwarmOptimizer;
  localBestCost: number = Number.POSITIVE_INFINITY;
  localBest: Vector;
  x: Vector;
  v: Vector;

  constructor(pso: ParticleSwarmOptimizer, x: Vector) {
    this.pso = pso;
    this.x = x;
    this.localBest = x.copy();
    this.v = new Vector(x.size());
  }

  update(r1: Vector, r2: Vector): void {
    this.v = this.v.multiplyScalar(this.pso.omega)
      .add(r1.multiplyScalar(this.pso.alpha1).multiply(this.localBest.subtract(this.x)))
      .add(r2.multiplyScalar(this.pso.alpha2).multiply(this.pso.globalBest.subtract(this.x)));
    this.x = this.x.add(this.v);
    const cost = this.pso.optimisable.cost(this.x);
    if (cost < this.localBestCost) {
      this.localBestCost = cost;
      this.localBest = this.x;
    }
    if (cost < this.pso.globalBestCost) {
      this.pso.globalBestCost = cost;
      this.pso.globalBest = this.x;
    }
  }
}

export class ParticleSwarmOptimizer {
  optimisable: Optimizable;
  particles: Particle[] = [];
  globalBest: Vector;
  globalBestCost: number = Number.POSITIVE_INFINITY;

  omega: number;
  alpha1: number;
  alpha2: number;
  
  constructor(optimizable: Optimizable, omega: number, alpha1: number, alpha2: number) {
    this.optimisable = optimizable;
    this.omega = omega;
    this.alpha1 = alpha1;
    this.alpha2 = alpha2;
    this.globalBest = optimizable.state().copy();
  }

  initParticles(n: number): void {
    this.particles = [];
    const size = this.globalBest.size();
    for (let i = 0; i < n; i++) {
      this.particles.push(new Particle(this, Vector.uniform(size)));
    }
  }

  update(): void {
    for (let particle of this.particles) {
      const r1 = Vector.uniform(particle.v.size());
      const r2 = Vector.uniform(particle.v.size());
      particle.update(r1, r2);
    }
  }
}

export class OptimizableNN implements Optimizable {
  net: nn.Node[][];
  loss: (net: nn.Node[][]) => number;

  constructor(net: nn.Node[][], loss: (net: nn.Node[][]) => number) {
    this.net = net;
    this.loss = loss;
  }

  cost(state: Vector): number {
    let i = 0;
    for (let layer of this.net) {
      for (let node of layer) {
        node.bias = state.values[i++];
        for (let link of node.inputLinks) {
          link.weight = state.values[i++];
        }
      }
    }
    return this.loss(this.net);
  }

  state(): Vector {
    let weights = new Vector(0);
    for (let layer of this.net) {
      for (let node of layer) {
        weights.values.push(node.bias);
        for (let link of node.inputLinks) {
          weights.values.push(link.weight);
        }
      }
    }
    return weights;
  }
}