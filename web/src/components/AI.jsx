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
  }

  async init() {
    try {
      console.log('Initializing AI with model URL:', this.modelUrl);
      
      // Configure onnxruntime-web to use CDN for WASM files
      // Try without specifying wasmPaths first - let it use bundled files
      // ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
      
      // Load the ONNX model in browser
      console.log('Creating inference session...');
      
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
      
      // Try to load the model with explicit external data configuration
      try {
        const externalDataUrl = `${this.modelUrl}.data`;
        console.log('Trying with external data:', externalDataUrl);
        
        this.session = await ort.InferenceSession.create(this.modelUrl, {
          executionProviders: ['wasm'],
          externalData: [{
            path: 'alphazero-network-model-onnx.onnx.data',
            data: externalDataUrl
          }]
        });
      } catch (externalDataError) {
        console.log('External data approach failed, trying without:', externalDataError.message);
        
        // Fallback: try without external data
        this.session = await ort.InferenceSession.create(this.modelUrl, {
          executionProviders: ['wasm']
        });
      }
      
      console.log('Model loaded successfully!');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);
      
      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];
      
      console.log('AI initialization complete');
    } catch (error) {
      console.error('AI initialization failed:', error);
      throw error;
    }
  }

  /**
   * Convert your 6x7 board to the 3-channel format the model expects
   * board: 2D array 6 rows x 7 columns
   */
  encodeBoard(board) {
    const channels = 3; // player1, empty, player2 (matching Python order)
    const height = 6;
    const width = 7;
    const tensorData = new Float32Array(channels * height * width);

    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const idx = r * width + c;
        const cell = board[r][c] || 0; // Convert null to 0
        
        // Match Python encode_state: (state == 1, state == 0, state == -1)
        // Channel 0: player 1 positions (cell === 1)
        tensorData[idx] = cell === 1 ? 1 : 0;
        
        // Channel 1: empty positions (cell === 0 or null)  
        tensorData[width*height + idx] = (cell === 0 || cell === null) ? 1 : 0;
        
        // Channel 2: player 2 positions (cell === 2, but Python uses -1)
        tensorData[2*width*height + idx] = cell === 2 ? 1 : 0;
      }
    }
    
    // Debug: Print the actual board values
    console.log("Board values:", board.map(row => row.map(cell => cell || 0)));
    
    return tensorData;
  }

  /**
   * Given a 2D board, compute the AI move
   */
  /*async getMove(board) {
    try {
      if (!this.session) throw new Error("Model not initialized");

      // The inputData should be a Flat32Array or similar typed array
      const inputData = this.encodeBoard(board);

      // inputShape should be an array representing the dimensions of the input
      const inputShape = [1, 3, 6, 7];

      // 2. Prepare the input tensor
      const inputTensor = new ort.Tensor('float32', new Float32Array(inputData), inputShape);
  
      // 3. Create the feeds object
      // The key (inputName) must match the input name defined in your ONNX model
      const feeds = { [this.inputName]: inputTensor };
  
      // 4. Run the inference session
      const results = await this.session.run(feeds);
  
       // 5. Extract the policy logits and select the best move
       console.log("*********** Results:", results);
       
       // Use the 'linear_1' output which contains the policy logits (7 values for 7 columns)
       const policyTensor = results['linear_1'];
       if (policyTensor && policyTensor.data.length === 7) {
         const policyLogits = Array.from(policyTensor.data);
         console.log("Policy logits:", policyLogits);
         
         // Find the column with the highest policy value
         const maxIndex = policyLogits.indexOf(Math.max(...policyLogits));
         console.log("Selected move:", maxIndex);
         return maxIndex; // should be a number between 0 and 6
       } else {
         throw new Error("Policy output not found or has wrong shape.");
       }
    } catch (e) {
      console.error(`Failed to inference ONNX model: ${e}`);
      throw e;
    }*/
    
    async getMove(board) {
      try {
        // The inputData should be a Flat32Array or similar typed array
        const inputData = this.encodeBoard(board);

        // inputShape should be an array representing the dimensions of the input
        const inputShape = [1, 3, 6, 7];

        // 2. Prepare the input tensor
        const inputTensor = new ort.Tensor('float32', new Float32Array(inputData), inputShape);
    
        // 3. Create the feeds object
        // The key (inputName) must match the input name defined in your ONNX model
        const feeds = { [this.inputName]: inputTensor };
    
        const results = await this.session.run(feeds);

        console.log('*********** Results:', results); // Log the actual results
    
        // Assume 'linear_1' (valueOutput) is the policy head because it has the correct shape.
        // And 'output' (policyOutput) is the value head.
        const policyOutput = results['207']; //results.linear_1;
        const valueOutput = results.output; 

      // Validate the policy output's shape before proceeding.
      // For a 7-column board, the policy should be a 1D tensor of size 7.
      if (!policyOutput || policyOutput.dims[1] !== 7) {
        throw new Error(`Policy output not found or has wrong shape. Found shape: ${policyOutput.dims}`);
      }

      const policyData = policyOutput.data;
      const bestMove = this.getBestMoveFromPolicy(policyData);

      return bestMove;
    } catch (e) {
      console.error('Failed to inference ONNX model:', e);
      throw e;
    }
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