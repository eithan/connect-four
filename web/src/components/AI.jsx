import * as ort from "onnxruntime-web";

/**
 * ONNX Connect Four AI
 */
class ConnectFourAI {
  constructor(modelUrl) {
    this.modelUrl = modelUrl;
    this.session = null;
    this.inputName = null;
    this.outputName = null;
    this.outputNames = [];
  }

  async init() {
    try {
      console.log('Initializing AI with model URL:', this.modelUrl);
      
      // Optional performance hints
      try {
        ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
        ort.env.wasm.simd = true;
      } catch (_) {}

      // First, verify the model file is accessible
      try {
        const response = await fetch(this.modelUrl);
        if (!response.ok) {
          throw new Error(`Model file not accessible: ${response.status} ${response.statusText}`);
        }
        console.log('Model file is accessible, size:', response.headers.get('content-length'), 'bytes');
      } catch (fetchError) {
        console.error('Failed to fetch model file:', fetchError);
        throw new Error(`Cannot access model file: ${fetchError.message}`);
      }

      // Try to load the model with different configurations
      let session = null;
      const configs = [
        // Try with external data first
        {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          externalData: [{
            path: 'alphazero-network-model.onnx.data',
            data: `${this.modelUrl}.data`
          }]
        },
        // Fallback without external data
        {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        },
        // Minimal config
        {
          executionProviders: ['wasm']
        }
      ];

      for (let i = 0; i < configs.length; i++) {
        try {
          console.log(`Trying ONNX config ${i + 1}/${configs.length}...`);
          session = await ort.InferenceSession.create(this.modelUrl, configs[i]);
          console.log(`Successfully loaded with config ${i + 1}`);
          break;
        } catch (configError) {
          console.log(`Config ${i + 1} failed:`, configError.message);
          if (i === configs.length - 1) {
            throw configError; // Re-throw the last error
          }
        }
      }

      if (!session) {
        throw new Error('All ONNX loading configurations failed');
      }

      this.session = session;
      this.inputName = this.session.inputNames[0];
      this.outputNames = this.session.outputNames;
      this.outputName = this.outputNames[0];
      
      console.log('AI initialization complete');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);
    } catch (error) {
      console.error('AI initialization failed:', error);
      throw error;
    }
  }

  /**
   * Convert your 6x7 board to the 3-channel format the model expects
   * board: 2D array 6 rows x 7 columns
   */
  encodeBoard(board, currentPlayer) {
    const channels = 3; // current player, empty, opponent (matching Python order)
    const height = 6;
    const width = 7;
    const tensorData = new Float32Array(channels * height * width);

    const opponentPlayer = currentPlayer === 1 ? 2 : 1;

    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const idx = r * width + c;
        const cell = board[r][c];

        const isEmpty = cell === null || cell === 0 || cell === undefined;
        const isCurrent = cell === currentPlayer;
        const isOpponent = cell === opponentPlayer;

        // Channel 0: current player positions (maps to Python state == 1)
        tensorData[idx] = isCurrent ? 1 : 0;
        // Channel 1: empty positions (maps to Python state == 0)
        tensorData[width * height + idx] = isEmpty ? 1 : 0;
        // Channel 2: opponent positions (maps to Python state == -1)
        tensorData[2 * width * height + idx] = isOpponent ? 1 : 0;
      }
    }
    return tensorData;
  }

  async getMove(board, currentPlayer) {
    try {
      if (!this.session) {
        throw new Error('AI session not initialized');
      }

      const inputData = this.encodeBoard(board, currentPlayer);
      const inputShape = [1, 3, 6, 7];
      const inputTensor = new ort.Tensor('float32', inputData, inputShape);
      const feeds = { [this.inputName]: inputTensor };

      const results = await this.session.run(feeds);

      // Identify policy and value outputs by shape
      let policyTensor = null;
      let valueTensor = null;
      for (const tensor of Object.values(results)) {
        const dims = tensor.dims || [];
        if (dims.length === 2 && dims[1] === 7) {
          policyTensor = tensor;
        } else if (dims.length === 2 && dims[1] === 1) {
          valueTensor = tensor;
        }
      }

      if (!policyTensor) {
        throw new Error('Policy output not found');
      }

      const logits = policyTensor.data; // Float32Array length 7

      // Mask invalid moves, then softmax
      const validColumns = this.getValidColumns(board);
      const masked = new Float32Array(7);
      for (let i = 0; i < 7; i++) {
        masked[i] = validColumns.includes(i) ? logits[i] : -1e9;
      }
      const probs = this.softmax(masked);

      // Choose argmax over valid columns
      let bestMove = -1;
      let bestProb = -1;
      for (let i = 0; i < 7; i++) {
        if (probs[i] > bestProb) {
          bestProb = probs[i];
          bestMove = i;
        }
      }

      // Fallback if all invalid (shouldn't happen)
      if (bestMove === -1) {
        bestMove = validColumns.length > 0 ? validColumns[0] : -1;
      }

      return bestMove;
    } catch (e) {
      console.error('Failed to inference ONNX model:', e);
      throw e;
    }
  }

  getValidColumns(board) {
    const cols = [];
    for (let c = 0; c < 7; c++) {
      const top = board[0][c];
      if (top === null || top === 0 || top === undefined) cols.push(c);
    }
    return cols;
  }

  softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) max = Math.max(max, arr[i]);
    let sum = 0;
    const exps = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      const v = Math.exp(arr[i] - max);
      exps[i] = v;
      sum += v;
    }
    for (let i = 0; i < arr.length; i++) exps[i] = exps[i] / (sum || 1);
    return exps;
  }

  getBestMoveFromPolicy(policyData) {
    let bestMove = -1;
    let maxProb = -1;

    for (let i = 0; i < policyData.length; i++) {
      if (policyData[i] > maxProb) {
        maxProb = policyData[i];
        bestMove = i;
      }
    }
    return bestMove;
  }
}

export default ConnectFourAI;