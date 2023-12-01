import subprocess
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# React code as a string
# react_files = {
#     "App.js": r"""
# import React from 'react';
# import './App.css';

# function App() {
#   return (
#     <div className="App">
#       <header className="App-header">
#         <p>
#           Hello, world!
#         </p>
#       </header>
#     </div>
#   );
# }

# export default App;
# """
# }

react_files2 = {
    "App.js": r"""
    import React from 'react';
import ReactDOM from 'react-dom';
import './index.css'; // Make sure to create an index.css file with the necessary styles

class HelloPage extends React.Component {
  render() {
    return (
      <div className="hello-world-container">
        Hello, world!
      </div>
    );
  }
}

ReactDOM.render(<HelloPage />, document.getElementById('root'));
""",
    "index.css": r"""
    .hello-world-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh; /* Use full viewport height */
  color: white; /* White text */
  background-color: #000; /* Black background */
  font-family: 'Arial', sans-serif; /* Assuming Arial font */
  font-size: 2em; /* Adjust as needed for your specific design */
}
""",
    "index.html": r"""
    <!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>React App</title>
</head>
<body>
  <div id="root"></div>
  <!-- The React script tags will be automatically injected here by your build tool -->
</body>
</html>
""",
}

# Step 1: Create a new React application
app_name = "my_react_app"
subprocess.call(["npx", "create-react-app", app_name])

# Step 2: Write the provided React code to App.js
for file_name, react_code in react_files2.items():
    with open(f"./{app_name}/src/{file_name}.js", "w") as file:
        file.write(react_code)

# Step 3: Run the React application in the background
# Execute in a terminal that stays open
# cd my_react_app
# npm start
proc = subprocess.Popen(
    ["npm", "start"],
    cwd=f"./{app_name}",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
time.sleep(5)  # Wait for the server to start

# Step 4: Take a screenshot using Selenium
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("http://localhost:3000")  # URL of your React app
driver.save_screenshot("react_app_2.png")
driver.quit()

# Step 5: Terminate the React app server
proc.terminate()
