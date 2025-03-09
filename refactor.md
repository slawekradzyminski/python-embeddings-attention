# Refactoring Plan for Python Sidecar

Below is a step-by-step, highly detailed plan for refactoring the current codebase to improve quality, thin out the API layer, move logging and dimension-reduction logic into their own modules, and split up the `/process` endpoint into three separate endpoints.

## 1. PREP WORK

### Create a new Git branch
- Name it something like `refactor/split_endpoints`
- This ensures your changes are isolated until you're ready to merge.

### Check your environment
- Ensure you have Python 3.12 (or your desired version) installed and that you can build/run Docker as before.

### Ensure your tests are passing initially
- Run `pytest` (or `python -m pytest`) in the project root.
- Fix any pre-existing test failures so you start from a clean baseline.

## 2. REFACTOR THE CODE STRUCTURE

### 2.1 Create a dedicated package for logging

#### Create app/logging_config.py
- This file will contain the logger configuration currently in main.py.
- Move all code that sets up the logger, console/file handlers, and log directories from main.py into app/logging_config.py.
- Provide a function (e.g., `setup_logger()`) that initializes and returns the logger.

#### Update app/main.py
- Remove all the logger configuration code and references (the rotating file handlers, etc.).
- At the top of main.py, import `setup_logger` from logging_config.
- Right after the global variables or the app creation, call `setup_logger()` so that the logger is fully configured before handling requests.
- Store the returned logger in a global variable or attach it to the FastAPI app as `app.state.logger` if you like.

#### Verification
- Confirm the logs still appear in both console and file.
- Run the app and check that the log directory and log file behave as before.
- Adjust pytest if any tests rely on previous logger references.

### 2.2 Extract dimension reduction logic to its own file/package

#### Create app/reduction_service.py (or similarly named)
- Copy the existing dimension-reduction code from model_service.py (dimensionality_reduction function) into a new class or set of functions in reduction_service.py.
- For example:

```python
class DimensionalityReducer:
    def __init__(self, method: str = "pca", n_components: int = 2):
        self.method = method
        self.n_components = n_components
    
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        # StandardScaler, PCA/UMAP, MinMaxScaler, etc.
        ...
        return reduced
```

- Keep it flexible, so you can instantiate with different method or n_components.

#### Update model_service.py
- Remove the old dimensionality_reduction method.
- If anywhere in model_service.py depends on that method, replace it with calls to the new DimensionalityReducer from reduction_service.py.

#### Adjust import statements
- Anywhere else in the code that references model_service.dimensionality_reduction, import and use DimensionalityReducer from reduction_service instead.

#### Verification
- Run all tests, especially the dimension-reduction tests. They should still pass unchanged or with minimal modifications.
- If you see any test referencing the old location, update it to reference the new module.

### 2.3 Make the API layer "thinner"

#### Create a separate "router" or "controller" file
- For example, app/routes.py or app/endpoints.py.
- Move all endpoint definitions (previously in main.py) to this file.
- Keep main.py as minimal as possible: just the app creation, middleware setup, logger setup, and app.include_router(...) calls if you use FastAPI's APIRouter.

```python
# app/routes.py
from fastapi import APIRouter, HTTPException
from .model_service import ModelService
from .logging_config import setup_logger
# etc.

router = APIRouter()

@router.post("/embeddings")
async def get_embeddings(...):
    ...
```

#### In main.py
- Keep only:
  - The FastAPI() instantiation
  - The logging setup
  - The middleware additions
  - app.include_router(router, prefix="") to bring in endpoints from routes.py
- Goal: main.py is mostly a "loader" that wires up the app, the router, and the logger. All real endpoint logic is in routes.py.

#### Verification
- Your tests that reference e.g. TestClient(app) should continue to work.
- No functionality should be lost; it's just a re-organization.

## 3. SPLIT THE /process ENDPOINT INTO THREE DISTINCT ENDPOINTS

The original `/process` currently returns tokens, embeddings, attentions, and optionally reduced embeddings. We will break that out:

### POST /embeddings

#### Request Body:
```json
{
  "text": "Hello world",
  "model_name": "gpt2"
}
```

#### Behavior:
- Load (or retrieve cached) ModelService for the requested model name.
- Run get_embeddings_and_attention(text) internally, but only return tokens and embeddings in the JSON.
- Do not return attention or dimension-reduced embeddings.

#### Response:
```json
{
  "tokens": [...],
  "embeddings": [...]
}
```
- If you want to still show which model was used, you can include "model_name" in the response.

### POST /attention

#### Request Body:
```json
{
  "text": "Hello world",
  "model_name": "gpt2"
}
```

#### Behavior:
- Retrieve the ModelService for model_name.
- Run the same forward pass to get tokens, hidden_states, attentions.
- Return only the attention data (and maybe the tokens if you need to label rows/cols in a heatmap).

#### Response:
```json
{
  "tokens": [...],
  "attention": [...]
}
```

Note: This is simpler to handle if you get the same outputs from model_service, but only include the relevant portion in the final JSON.

### POST /reduce

#### Request Body:
```json
{
  "text": "Hello world",
  "model_name": "gpt2",
  "n_components": 2,
  "reduction_method": "pca"
}
```

#### Behavior:
- Retrieve the ModelService.
- Get the embeddings from model_service.get_embeddings_and_attention(...).
- Call the new DimensionalityReducer from reduction_service.py to reduce the embeddings to the specified dimension.
- Return the tokens plus the reduced embeddings (so your front-end or other code can label them).

#### Response:
```json
{
  "tokens": [...],
  "reduced_embeddings": [...]
}
```

### Remove or deprecate the old /process endpoint entirely
- Or if you want to keep it for backward compatibility, mark it as deprecated and do not advertise it.

### Implementation Hints:
- You do not strictly need to run the forward pass 3 separate times for each type of data, but the simplest approach for clarity is "each endpoint calls model_service.get_embeddings_and_attention and returns partial results." (Performance might be slightly lower, but code clarity is higher. If you want advanced caching, you can do that later.)
- Each endpoint can create a distinct Pydantic model in app/routes.py or in a separate schemas.py file, if you want to keep it extra tidy.

### Verification
- Write or update unit tests in test_api.py or create new test files specifically for these endpoints:
  - test_endpoint_embeddings.py tests the /embeddings endpoint.
  - test_endpoint_attention.py tests the /attention endpoint.
  - test_endpoint_reduce.py tests the /reduce endpoint.
- Ensure each test verifies only the data it expects (no huge embedding data from /attention or vice versa).

## 4. PYTEST COVERAGE FOR ALL CHANGES

You already have a robust test suite in tests/. Update or add new tests to handle the new endpoints and the newly extracted logic:

### Test for app/logging_config.py
- If the AI can do advanced reflection, you can confirm the logger is set up with file handlers, etc.
- More simply, just ensure the refactor didn't break existing logs.

### Test for dimension reducer (reduction_service.py)
- Move or copy the dimension-reduction tests from test_model_service.py into test_reduction_service.py.
- Thoroughly test PCA with 2D, 3D, and also UMAP if you use that.
- Confirm the shape of the returned array, plus any scaling transformations.

### Test for new endpoints
- test_embeddings_endpoint
  - Input a known short text.
  - Check that the response includes tokens and embeddings.
  - Check that attention is not in the response.
- test_attention_endpoint
  - Input the same short text.
  - Check that the response includes tokens and attention but does not include embeddings.
- test_reduce_endpoint
  - Input the same short text, request 2D reduction with PCA.
  - Check that the response includes tokens and reduced_embeddings.
  - Confirm len(reduced_embeddings) == len(tokens) and each vector has length 2.

### Ensure all tests pass
- The final step once you've done the refactoring is pytest again.
- If coverage is important, run `coverage run -m pytest` or similar and check coverage.
- Fix any errors or coverage holes.

## 5. ADDITIONAL (OPTIONAL) IMPROVEMENTS

Below are extra ideas to further refine the code:

### Explicit model-caching
- For each model name, store references in a dictionary to avoid reloading. (You already do something similar in model_cache.)

### Async vs Sync
- FastAPI supports async endpoints. If performance or concurrency is a concern, ensure your model calls use non-blocking strategies if feasible.
- For CPU-bound PyTorch operations, an event loop won't help that much, but it can help keep your server responsive.

### Timeouts or concurrency limits
- If the calls might be slow (large models, big texts), you might consider concurrency-limits or request timeouts to keep your service stable.

### Add typing throughout
- Add type hints to every function signature, especially for things that return complex dictionaries or arrays.

### Refine Docker setup
- If your memory usage is large, set explicit resource constraints in docker-compose.yml.
- If you want GPU usage, change the base image to something like nvidia/cuda.

### Security
- If you plan to open this service publicly, consider adding auth or at least an API token check.

## 6. MERGE & CLEAN UP

### Merge or create a PR
- Once you validate everything works, open a pull request from refactor/split_endpoints to main (or your main branch).

### Document
- Update README.md so it reflects the new endpoints (/embeddings, /attention, /reduce).
- Possibly remove references to the old /process or mark them as deprecated.

### Deployment
- Rebuild Docker container:
```bash
docker build -t python-sidecar:latest .
docker run -p 5000:5000 python-sidecar:latest
```
- Or `docker-compose up -d` after verifying the updated docker-compose.yml is correct.

### Final Testing
- Check logs, container status, and confirm the new endpoints respond as expected.
- Validate the improved code structure (thinner main.py, separate modules for logging & reduction) and the simpler payloads from each endpoint.

## 7. FINAL RECAP

Following these instructions will:
- Split /process into three smaller, single-purpose endpoints.
- Move logger setup to app/logging_config.py.
- Extract dimension reduction logic into app/reduction_service.py.
- Keep each piece of the system more maintainable and testable.
- Provide pytest coverage for each new endpoint and new module.

With this plan, you will have a more modular codebase that follows Python best practices. Each step includes the exact refactoring tasks, verifying them via pytest, and culminating in a stable, well-organized service.