**supereeg**: A Python toolbox for inferring whole-brain activity from sparse ECoG recordings
==================================================================================================

.. image:: _static/example_model.gif
    :width: 400pt

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorial
   api
   auto_examples/index
   migration

`supereeg <https://github.com/ContextLab/supereeg>`_ (name inspired by Robert Sawyer's `The Terminal Experiment <https://en.wikipedia.org/wiki/The_Terminal_Experiment>`_) is a (fictional) tool for recording the electrical activities of every neuron in the living human brain.  Our approach is somewhat less ambitious, but (we think) still "super" cool: obtain high spatiotemporal estimates of activity patterns throughout the brain using data from a small(ish) number of  `implanted electrodes <https://en.wikipedia.org/wiki/Electrocorticography>`_.  The toolbox is designed to analyze ECoG (electrocorticographic) data, e.g. from epilepsy patients undergoing pre-surgical evaluation.

The way the technique works is to leverage data from different patients' brains (who had electrodes implanted in different locations) to learn a "correlation model" that describes how activity patterns at different locations throughout the brain relate.  Given this model, along with data from a sparse set of locations, we use Gaussian process regression to "fill in" what the patients' brains were "most probably" doing when those recordings were taken.  Details on our approach may be found in `this preprint <http://biorxiv.org/content/early/2017/03/27/121020>`_.  You may also be interested in watching `this talk <https://www.youtube.com/watch?v=t6snLszEneA&feature=youtu.be&t=35>`_ or reading this `blog post <https://community.sfn.org/t/supereeg-ecog-data-breaks-free-from-electrodes/8344>`_ from a recent conference.

Although our toolbox is designed with ECoG data in mind, in theory this tool could be applied to a very general set of applications.  The general problem we solve is: given known (correlational) structure of a large number of "features," and given that (at any one time) you only observe some of those features, how much can you infer about what the remaining features are doing?
