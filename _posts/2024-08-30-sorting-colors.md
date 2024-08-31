My friend sent me a meme about using your favorite pokemon to style your plots and I was instantly nerd snipe.
"That sounds easy and fun to automate. I should create a python package for it."
Oh how naive I was.
Most of it was easy, and the final result is great (shameless plug for palettetown).

https://github.com/rowland-208/palettetown/blob/main/etc/examples.ipynb

But there was one problem I did not see coming that took a bit of creativity to solve.
The end result was a python package called colorsort that sorts colors.
You can see it in action here.

https://github.com/rowland-208/colorsort/blob/main/etc/examples.ipynb

Sorting colors doesn't sound so hard at first, but when you stop and think about it what does it mean for one color to be less than another?
For palettetown the big challenge is to pick colors and order them such that:
* the resulting plots are reminiscent of the pokemon,
* the plots are perceptually monotonic and continuous for false color plots, and
* curves are easy to distinguish for line and scatter plots.

In my early investigation I found several ways of sorting colors for false color plots.
I found a great resource with some suggestions:

https://www.alanzucconi.com/2015/09/30/colour-sorting/

The ideas explored include:
* RGB sort (terible)
* HSV sort (bad)
* Luminosity sort (bad)
* Step sorting (good)
* Hilbert sort (weird)
* Traveling salesman sort (weird)

Along the way I learned about color spaces and color distances. The OpenCV docs have an excellent readme on the color spaces defined in that library and the transformations between them. The LAB color space is particularly interesting. It was designed to be perceptually uniform with a Euclidean distance (also called CIELAB and the CIE76 distance). The smallest perceptible CIE76 distance is around 2.

https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

The CIE76 distance was shown to have flaws and subsequent color distances address these. However, these distances are not Euclidean so I find them difficult to reason about.

https://en.wikipedia.org/wiki/Color_difference

To work with the CIEDE2000 metric (the current state of the art as far as I can tell) I found the colormath library to be useful. Users on some forums complained that it is slow.

https://pypi.org/project/colormath/

Once I made a CIEDE2000 distance matrix between colors I ran into difficulty in deciding my goal, e.g., in designing a cost function. One condition we want is to minimize the nearest neighbor distance. I started looking into algorithms for association problems because I’m using those at work. I found that scipy has an implementation of the Hungarian algorithm to maximize an association matrix.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

I realized that linear sum assignment cannot be applied here. This led me to the travelling salesman problem. There is also the or-tools library to solve TSPs. I didn’t use these libraries, but they seem like they would be useful.

https://pypi.org/project/ortools/
https://developers.google.com/optimization/routing/tsp#python

Next I looked into clustering algorithms. We sort of want to cluster the colors so that nearest neighbors are similar. The TSP would achieve this, but ward clustering works too. I learned about some of the clustering functions in scipy. The fcluster method applies a closeness criteria (or other criterion) to produce clusters.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster

I also learned that the most recent version of the ward clustering does not accept a full density matrix. A condensed density matrix must be supplied. This can be converted from a full density matrix using.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

I also found the leaders function which selects the observation closest to the top of a dendrogram. I didn’t have luck using this one.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaders.html#scipy.cluster.hierarchy.leaders

I found the optimal_leaf_ordering function which reorganizes leaves so that nearest neighbors have the smallest distance.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering

Nexted I started thinking along the lines of constructing a curve through LAB space. We want the curve to start at min luminosity, and end at max luminosity. It should increase monotonically. And it should pass through the desired colors, or close to them. The problem here is that unwanted colors may be picked up along the way, e.g., if going from blue to red you would pick up purple. Anyway, I would need to calculate the closest point to a polynomial and found a good post about that.

https://stackoverflow.com/questions/2742610/closest-point-on-a-cubic-bezier-curve

Another idea I had that is less well formed is to partition LAB space and choose a representative color from each. That is basically the same as clustering though. So maybe K-means would work.

Things I tried that didn’t work well:
* Sorting tuples in LAB, HSV
* Step sorting
* Ward clustering

Current approach:
* If more than 10 colors, reduce to 10 by k-means clustering
* Solve travelling salesman and force highest and lowest luminance points to be neighbors (https://stackoverflow.com/questions/14527815/how-to-fix-the-start-and-end-points-in-travelling-salesmen-problem)

It worked! I ended up using k means to select 10 colors. Then travelling salesman with the CIE2000 distance function. I ended up scaling the luminance by 1/10 to reduce the impact on sorting. Runs of colors are more important than monotonically increasing luminance. You can see results in this jupyter notebook.

![alt text](_assets/saturn.png)

![alt text](_assets/saturn-colors.png)

![alt text](_assets/parrots.png)

![alt text](_assets/parrots-colors.png)

Finally, as a separate endeavor I learned about creating a python package. I found this article that was helpful, except that the section on including data has a poor recommendation. Instead of pkg_resource it is advisable to use the pkgutil.get_data method in the standard library. I learned how to use tox. It handles the environment side of testing, including testing multiple python distros.

https://kiwidamien.github.io/making-a-python-package-viii-summary.html
https://tox.readthedocs.io/en/latest/example/basic.html

The python package file structure I recommend is

packagename/
	packagename/
		data/
			somedata.json
		tests/
			test_module.py
		module.py
	etc/
		examples.ipynb
	bin/
		script.py
	setup.py
	README.md
	LICENSE
	tox.ini
	.gitignore

And the license I recommend is MIT for most things. The GNU license is way too intense for my personal projects. I want people to actually use what I’m making so MIT or BSD are good there. And I like MIT over BSD for aesthetic reasons.

https://www.youtube.com/watch?v=DDx6gjwU0K8
https://opensource.stackexchange.com/questions/217/what-are-the-essential-differences-between-the-bsd-and-mit-licences



