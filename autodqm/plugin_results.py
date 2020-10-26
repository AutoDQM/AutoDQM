#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class PluginResults(object):
    """Data class for storing the results of a plugin function

    self.canvas: the canvas to be saved and displayed
    self.show: whether the canvas should be shown by default
    self.info: dictionary of any extra information that should be displayed
    self.artifacts: root objects that need to be protected from garbage collection
    """

    def __init__(self, canvas, show=False, info={}, artifacts=[]):
        self.canvas = canvas
        self.show = show
        self.info = info
        self.root_artifacts = artifacts
