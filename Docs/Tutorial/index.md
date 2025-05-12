# Tutorial
This section will establish an understanding of the pre-requisite resources and 
basic workflow for applications created with the De-Orbiting Satellite Tracking 
System (D-OSTS).

## Quickstart Guide
### Installation
First, after downloading the source code from the D-OSTS GitHub Repository, ensure 
build tools is installed on your local machine. You can easily do this with 
`pip install`.

```bash
pip install build
````

You can then create a `dist/` folder with `.whl` and `.tar.gz` files by using these 
new build tools.

```bash
python -m build
```

Once again, using `pip install`, we locally install D-OSTS.

```bash
pip install dist/dosts-1.0.0-py3-none-any.whl
```

And, that's it, you're ready to start!

### Dependencies
### Environment
Test the environment to see if D-OSTS has installed correctly.

From the interactive shell, this is done simply.

```python
>>> import sys
>>> 'dosts' in sys.modules
True
```

Alternatively, successful installation can also be tested in-script.

```python
try:
    import dosts
    print("dosts imported successfully")
except ImportError:
    print("dosts not installed or failed to import")

```

### License
MIT License

Copyright (c) 2025 Jai Bharaj, Tom Miller, David Dukov, Linan Yang, Kunlin Cai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Simulation
### True Trajectory
### Noisy Measurements
### Full Simulation Build

## Prediction
### Initialisation
### Extended Kalman Filters
### Full Prediction Build

## Visualisation
