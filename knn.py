"""
    File:        knn.py
    Author:      Eva Nolan
    Course:      CS 307 - Computational Geometry
    Assignment:  Synthesis Project
    SOURCES: nearest neighbor algorthm performs the tasks outlined in lecture
    notes by Oregon State University Professor Thinh Nguyen:
    http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf
"""
from typing import List, Tuple
import time
import matplotlib.pyplot as plt
import random
import math
import datetime
import csv

NUM_ITERATIONS = 3
NODES_VISITED = 0

# node class ===================================================================
class Node:
    def __init__(self, value, left=None, right=None, region=None, parent=None, point=None):
        self.left = left
        self.right = right
        self.value = value
        self.region = region
        self.parent = parent
        self.point = point

    def set_left(self, node):
        self.left = node

    def set_right(self, node):
        self.right = node

    def set_region(self, region):
        self.region = region

    def set_parent(self, node):
        self.parent = node

    def set_point(self, pt):
        self.point = pt

    def is_leaf(self):
        return self.right == None and self.left == None

    def depth_print(self, depth):
        """ adapted from tutorialspoint.com """
        if self.right:
           self.right.depth_print(depth+1)
        print(" "*5*depth, self.value),
        if self.left:
           self.left.depth_print(depth+1 )

    def print(self):
        self.depth_print(0)


# create_kdtree and its helper function ========================================
def create_kdtree_helper(xsorted, ysorted, depth, region):
    """recursive function to generate kd tree """
    if len(xsorted) == 0:
        return None
    if len(xsorted) == 1:
        node = Node(xsorted[0])
        node.set_region(region)
        node.set_point(node.value)
        return node

    (lregion, rregion) = (region.copy(), region.copy())
    if depth%2 == 0:
        #even depth, split x
        median_loc = (len(xsorted)-1)//2
        median_pt = xsorted[median_loc]
        median = median_pt[0]
        # update regions
        (lregion[1], rregion[0]) = (median, median)
        leftx = xsorted[:median_loc+1]
        rightx = xsorted[median_loc+1:]
        (lefty, righty) = ([],[])
        for point in ysorted:
            if point[0]<=median:
                lefty.append(point)
            else:
                righty.append(point)
    else:
        #odd depth, split y
        median_loc = (len(ysorted)-1)//2
        median_pt = ysorted[median_loc]
        median = median_pt[1]
        #update regions
        lregion[3] = median
        rregion[2] = median
        lefty = ysorted[:median_loc+1]
        righty = ysorted[median_loc+1:]
        (leftx, rightx) = ([],[])
        for point in xsorted:
            if point[1]<=median:
                leftx.append(point)
            else:
                rightx.append(point)

    left_sub = create_kdtree_helper(leftx, lefty, depth + 1, lregion)
    right_sub = create_kdtree_helper(rightx, righty, depth +1, rregion)
    v = Node(median, left_sub, right_sub, region, None, median_pt)
    left_sub.set_parent(v)
    right_sub.set_parent(v)
    return v

def create_kdtree(points: List[Tuple[int,int]]):
    """
    Construct a kd-tree from a list of 2D points
    and return its root.

        Keyword arguments:
        points -- the list of 2d points

    Return the root of the kd-tree
    """
    # sort points by x
    xsorted = sorted(points)
    (xmin, xmax) = (xsorted[0][0], xsorted[-1][0])
    # sort points by y
    ysorted = sorted(points, key=lambda pt: pt[1])
    (ymin, ymax) = (ysorted[0][1], ysorted[-1][1])
    region = [xmin, xmax, ymin, ymax]
    return create_kdtree_helper(xsorted, ysorted, 0, region)


# range_query & its helper functions ===========================================
def in_R(range, point):
    """ Return True if point is in query range"""
    (xmin, xmax) = range[0]
    (ymin, ymax) = range[1]
    x = point[0]
    y = point[1]
    return xmin < x and x < xmax and ymin < y and y < ymax

def orient(o,t,pt):
    """returns the deteminant of matrix [[1,px,py],[1,qx,qy][1,rx,ry]]
       p = origin, q = target, r = point"""
    det = (t[0]*pt[1] + o[0]*t[1] + pt[0]*o[1])-(t[0]*o[1] + pt[0]*t[1] +
           pt[1]*o[0])
    if det == 0:
        return 0
    else:
        return det / abs(det)

def intersect(seg1, seg2):
    """
    returns True if segment 1 intersects segment 2
    """
    if orient(seg1[0], seg1[1], seg2[0]) != orient(seg1[0], seg1[1], seg2[1]):
        return orient(seg2[0], seg2[1], seg1[0]) != orient(seg2[0], seg2[1], seg1[1])
    return False

def overlap(r1, r2):
    """
    returns True if an edge of rectangle 1 intersects rectangle 2 or vice versa.
    Returns False otherwise.
    """
    r1_pts = [(r1[0], r1[2]), (r1[1], r1[2]), (r1[1], r1[3]), (r1[0], r1[3]), (r1[0], r1[2])]
    r2_pts = [(r2[0], r2[2]), (r2[1], r2[2]), (r2[1], r2[3]), (r2[0], r2[3]), (r2[0], r2[2])]

    for i in range(len(r1_pts)-1):
        seg1 = (r1_pts[i], r1_pts[i+1])
        for j in range(len(r1_pts)-1):
            seg2 = (r2_pts[j], r2_pts[j+1])
            # return true if intersection is found
            return intersect(seg1, seg2)
    return False

def sides_overlap(rectangle, container):
    """
    takes a rectangle and a container region, each a list of form
    [xmin, xmax, ymin, ymax]. Returns the number of the rectangle's corners that
    are inside the container. If returns 4, the rectangle is entirely inside
    the container. If 0, the rectangle is outside the container.
    """
    sum = 0
    (cxmin, cxmax) = (container[0], container[1])
    (cymin, cymax) = (container[2], container[3])

    corners = [(rectangle[0], rectangle[2]), (rectangle[0], rectangle[3]),
                (rectangle[1], rectangle[2]), (rectangle[1], rectangle[3])]
    for corner in corners:
        if cxmin < corner[0] and corner[0] < cxmax and cymin < corner[1] and corner[1] < cymax:
            # the x & y coordinates of the rectangle are inside the container
            sum += 1
    if sum == 4:
        # entire rect contained in container.
        return 1
    elif overlap(container, rectangle):
        # partial overlap
        return 0
    else:
        # no overlap
        return -1

def report_subtree(node):
    """
    reports all leaves in the tree with given root node
    """
    if node.is_leaf():
        return [node.value]
    pts_in_R = []
    if node.left:
        for pt in report_subtree(node.left):
            pts_in_R.append(pt)
    if node.right:
        for pt in report_subtree(node.right):
            pts_in_R.append(pt)
    return pts_in_R

def range_query(kd_tree, query_range: Tuple[Tuple[int,int],Tuple[int,int]]) -> List[Tuple[int,int]]:
    """
    Perform a 2D range reporting query using kd_tree and the given query range
    and return the list of points.

        Keyword arguments:
        kd_tree: the root node of the kd-tree to query
        query_range: a rectangular range to query

    Return the points in the query range as a list of tuples.
    """
    report = []
    (xmin, xmax) = query_range[0]
    (ymin, ymax) = query_range[1]
    if kd_tree.is_leaf():
        if in_R(query_range, kd_tree.value):
            report.append(kd_tree.value)
    else:
        # check left subtree
        insides = sides_overlap(kd_tree.left.region, [xmin, xmax, ymin, ymax])
        if insides == 1:
            # query region entirely contains tree region
            for pt in report_subtree(kd_tree.left):
                report.append(pt)
        elif insides == 0:
            # tree region overlaps query region
            for pt in range_query(kd_tree.left, query_range):
                report.append(pt)
        # now check right subtree
        insides = sides_overlap(kd_tree.right.region, [xmin, xmax, ymin, ymax])
        if insides == 1:
            # query region entirely contains tree region
            for pt in report_subtree(kd_tree.right):
                report.append(pt)
        elif insides == 0:
            # tree region overlaps query region
            for pt in  range_query(kd_tree.right, query_range):
                report.append(pt)
    return report


# nearest neighbor query functions =============================================
#new
def dist(pt_a, pt_b):
    """ returns the distance between two points """
    return math.sqrt((pt_a[0]-pt_b[0])**2 + (pt_a[1]-pt_b[1])**2)

def nearest_neighbor_helper(node, target, curr_nearest, curr_dist, depth):
    """
    recursive helper function returns the nearest neaighbor to a target point,
    given the current nearest neighbor and its distance from target, the root of
    the tree we are searching, and the depth of the overall tree where the
    subtree is found.
    """
    global NODES_VISITED
    NODES_VISITED += 1

    depth += 1
    if node.left == None and node.right == None:
        # leaf
        new_dist = dist(target, node.point)
        if new_dist < curr_dist:
            curr_dist = new_dist
            curr_nearest = node.point
        return curr_nearest

    else:
        point = node.point
        new_dist = dist(target, point)
        # update new closest point if neccessary
        if curr_dist == None or new_dist < curr_dist:
            curr_dist = new_dist
            curr_nearest = point

        # choose side we are on:
        region = node.right.region
        xrange = (region[0], region[1])
        yrange = (region[2], region[3])

        # dist from median line:
        if depth % 2 == 0:
            # even depth => splitting on x
            line_dist = abs(target[0] - point[0])
        else:
            line_dist = abs(target[1] - point[1])
        # other_pt is set only if closer points can be found on "wrong" side of tree
        other_pt = None
        if in_R((xrange, yrange), target):
            # traverse right
            best_pt = nearest_neighbor_helper(node.right, target, curr_nearest, curr_dist, depth)
            if dist(best_pt, target) > line_dist:
                # must traverse other side of tree
                other_pt = nearest_neighbor_helper(node.left, target, curr_nearest, curr_dist, depth)
        else:
            #traverse left
            best_pt =  nearest_neighbor_helper(node.left, target, curr_nearest, curr_dist, depth)
            if dist(best_pt, target) > line_dist:
                # must traverse other side of tree
                other_pt = nearest_neighbor_helper(node.right, target, curr_nearest, curr_dist, depth)
        if other_pt != None and dist(best_pt, target) > dist(other_pt, target):
            return other_pt
        return best_pt

def nearest_neighbor(kd_tree, target):
    """ calls recursive helper function to search for nearest neighbor given
    target node and a kd tree"""
    global NODES_VISITED
    NODES_VISITED = 0
    return nearest_neighbor_helper(kd_tree, target, None, float('inf'), -1)

# helper functions for testing =================================================
def generate_points(n):
    """
    randomly generates n points that are not colinear
    """
    (xlist, ylist) = ([],[])
    points = []
    for i in range(n):
        xlist.append(i)
        ylist.append(i)
    random.shuffle(ylist)
    for i in range(n):
        points.append((xlist[i], ylist[i]))
    return points

def generate_range(n, size):
    """
    Generates a query range to go along with generate_points. Returns a list
    of x & y bounds: [xmin, xmax, ymin, ymax]
    Keeps k fairly contsant.
    """
    size = size//2
    min = n//2 - size
    if min < 0:
        min = 0
    max = n//2 + size
    return ((min, max), (min, max))

def check_closest(kd_tree, target, neighbor):
    """ check correctness of knn search output """
    # get dist between target and output neighbor
    distance = dist(target, neighbor)
    xmin = target[0] - distance
    xmax = target[0] + distance
    ymin = target[1] - distance
    ymax = target[1] + distance
    query_range = ((xmin, xmax), (ymin, ymax))
    pts_in_range = range_query(kd_tree, query_range)
    for pt in pts_in_range:
        return dist(target, pt) > distance
    return True

def test():
    """
    Test nearest neighbor query.
    """

    n = int(input("Enter desired number of points "))
    points = generate_points(n-1)
    tree = create_kdtree(points)
    region = tree.region
    # print(region)
    target = (random.randint(region[0], region[1]), random.randint(region[2], region[3]))
    result = nearest_neighbor(tree, target)

    for i in range(len(points)):
        plt.scatter(points[i][0], points[i][1], color = "black")
    plt.scatter(target[0], target[1], color = "blue")
    plt.scatter(result[0], result[1], color = "red")
    plt.show()

def time_test():
    """
    Runs the nearest neighbor query, timing experiements as n doubles. Outputs
    the average number of nodes visited for a given input size and the total
    time taken to query on 100 experiments per size.
    """
    #set up csv to store runtime data
    dfile = open("nndata.csv", "w+" )
    dfile = open("nndata.csv", "a", newline = "" )
    writer = csv.writer(dfile)

    writer = csv.writer(dfile)
    writer.writerow(("n", "nodes visited", "runtime (ms)"))
    #reccomend 13 experiements. More takes too long to build tree.
    k = int(input("number of tests: "))
    print("{:>15} ".format("n"), end="")
    print("{:>25} ".format("nodes visited"), end="")
    print("{:>25} ".format("time (microseconds)"))
    for test_num in range(k):
        n = 2**test_num
        #get average time and num nodes over 8 runs
        total_time = 0
        num_success = 0
        total_nodes_visited = 0
        for sample in range(100):
            points = generate_points(n)
            tree = create_kdtree(points)
            #time query
            region = tree.region
            start = datetime.datetime.now()
            target = (random.randint(region[0], region[1]), random.randint(region[2], region[3]))
            end = datetime.datetime.now()
            time_diff = end - start
            time_ms = time_diff.microseconds

            total_time += time_ms
            result = nearest_neighbor(tree, target)
            if check_closest(tree, target, result):
                num_success += 1
                total_nodes_visited += NODES_VISITED

        #calculate and store average time & nodes visited
        avg_time = total_time / num_success
        avg_time = total_time
        avg_nodes_visted = total_nodes_visited / num_success
        print("{:>15} ".format(n), end="")
        print("{:>25} ".format(avg_nodes_visted), end="")
        print("{:>25} ".format(avg_time))
        row = [n, avg_nodes_visted, avg_time]
        writer.writerow(tuple(row))

    show = input("Would you like a visual example? Y/N ")
    if show == "Y" or show == "y":
        test()
    else:
        return


if __name__ == "__main__":
    time_test()
    # test() # use for testing, comment when done
