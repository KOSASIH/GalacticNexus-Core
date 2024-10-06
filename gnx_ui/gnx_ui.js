// Import the necessary libraries
import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter, Route, Switch } from 'react-router-dom';
import { Provider } from 'react-redux';
import store from './store';
import App from './App';
import Login from './Login';
import Register from './Register';
import Protected from './Protected';

// Define the gnx_ui parameters
const gnxUiParams = {
  // Define the gnx_ui parameters here
};

// Define the gnx_ui function
function gnxUi() {
  // Implement the gnx_ui function here
  return (
    <Provider store={store}>
      <BrowserRouter>
        <Switch>
          <Route path="/" exact component={App} />
          <Route path="/login" component={Login} />
          <Route path="/register" component={Register} />
          <Route path="/protected" component={Protected} />
        </Switch>
      </BrowserRouter>
    </Provider>
  );
}

// Render the gnx_ui function
ReactDOM.render(gnxUi(), document.getElementById('root'));
