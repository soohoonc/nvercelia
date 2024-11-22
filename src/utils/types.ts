export interface GraphNode {
  id: string;
  name: string;
  val: number;
  group: 'input' | 'hidden' | 'output';
  layer: number;
}

export interface GraphLink {
  source: string;
  target: string;
  value: number;
  weight: number;
} 