---
layout: post
title: Power Graph
date: 2021-01-17 12:00:00
---

The following is a first pass at graphing more complex inputs:

{% include power-graph.html %}

To create this graph, we take into account the following JavaScript
learnings:

- Instantiate variables as either changing (`var`) or constant (`const`)
- Variables do not need to be hard-typed, as JavaScript is a dynamic language
- Similar to a Lambda expression in Python, arrays can be iterated over in a single line using `map` and the arrow operator (`=>`)
- JavaScript has a built in `Math` class with methods such as `ceil` or `max`
- To debug directly to the HTML output...
    - You can create an HTML component, label it as "debug", and write to it within the JS code
    - You can add the output to an alert
    - You can write to the console, which can be pulled up in "Inspector" in most browsers
- The spread operator `...` allows you to explode an array
- Arrays start at 0 and can be indexed the same as in Python
- If-else statements are the same as in Java, with comparisons such as `==`, `>`, etc.
- For-loop consists of `(i=0; i<20; i++)` or `(index in [0.1, 0.2])`

```js
// ... <-- Within HTML components --> ...
// <p id="debug"></p>

const stats_distrib_value = 1.96;
var graph_param = 0.2;

var x_values = [0.1, 0.2, 0.3, 0.4, 0.5];
var transform_x = x_values.map(x_value => x_value*15 - stats_distrib_value);

// Three ways to debug in JavaScript
document.getElementById("debug").innerHTML = Math.max(...x_values);
alert(Math.max(...x_values));
console.log(Math.max(...x_values));
```

HTML learnings:

- Input components can be used to pull user input
- Input component of type `radio` are radio buttons that can be `checked` to show true or false
- Name the `radio` input components the same to treat them as a connected unit
- The `range` input allows you to create a slider with inputs `min`, `max`, and `step`
- Input component of a data type (e.g., `text` or `float`) creates a user-defined text box for input
- Placeholder can be used to show example input in the text box type
- Any attribute specified in the HTML component (e.g., `value`, `id`, or `checked`) can be referenced from the component in JS using JSON notation: `component.value`

Resources:

- [Material Design Palette](https://www.materialpalette.com/)