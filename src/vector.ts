export class Vector {
  values: number[] = [];

  constructor(size: number) {
    for (let i = 0; i < size; i++) {
      this.values.push(0);
    }
  }

  size(): number { return this.values.length; }

  copy(): Vector {
    let res = new Vector(this.values.length);
    for (let i = 0; i < this.values.length; i++) {
      res.values[i] = this.values[i];
    }
    return res;
  }

  add(other: Vector): Vector {
    if (this.size() != other.size()) {
      throw new RangeError();
    }
    let res = new Vector(0);
    for (let i = 0; i < this.values.length; i++) {
      res.values.push(this.values[i] + other.values[i]);
    }
    return res;
  }

  subtract(other: Vector): Vector {
    if (this.size() != other.size()) {
      throw new RangeError();
    }
    let res = new Vector(0);
    for (let i = 0; i < this.values.length; i++) {
      res.values.push(this.values[i] - other.values[i]);
    }
    return res;
  }

  multiply(other: Vector): Vector {
    if (this.size() != other.size()) {
      throw new RangeError();
    }
    let res = new Vector(0);
    for (let i = 0; i < this.values.length; i++) {
      res.values.push(this.values[i] * other.values[i]);
    }
    return res;
  }

  multiplyScalar(other: number): Vector {
    let res = new Vector(0);
    for (let i = 0; i < this.values.length; i++) {
      res.values.push(this.values[i] * other);
    }
    return res;
  }

  dot(other: Vector): number {
    if (this.size() != other.size()) {
      throw new RangeError();
    }
    let res = 0.0;
    for (let i = 0; i < this.values.length; i++) {
      res += this.values[i] * other.values[i];
    }
    return res;
  }

  static uniform(size: number): Vector {
    let res = new Vector(0);
    for (let i = 0; i < size; i++) {
      res.values.push(Math.random());
    }
    return res;
  }
}
