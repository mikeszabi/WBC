# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:21:47 2017

@author: SzMike
"""

#!/usr/bin/env python
# -*- coding: utf8 -*-
import _init_path
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


class AnnotationWriter:

    def __init__(self, folder, filename, imgagesize, author='Unknown'):
        self.folder = folder
        self.filename = filename
        self.author = author
        self.imgagesize = imgagesize
        self.objectlist = []

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.folder is None or \
                self.imgagesize is None or \
                len(self.objectlist) <= 0:
            return None

        top = Element('annotation')

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        folder = SubElement(top, 'folder')
        folder.text = self.folder

        source = SubElement(top, 'source')
        author = SubElement(source, 'submittedBy')
        author.text = self.author

        imagesize = SubElement(top, 'imagesize')
        nrows = SubElement(imagesize, 'nrows')
        ncols = SubElement(imagesize, 'ncols')
        nrows.text = str(self.imagesize[0])
        ncols.text = str(self.imagesize[1])
        return top

#    def addBndBox(self, xmin, ymin, xmax, ymax, name):
#        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
#        bndbox['name'] = name
#        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.objectlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = each_object[0]     
            p_type = SubElement(object_item, 'type')
            p_type.text = each_object[1]     
            polygon = SubElement(object_item, 'polygon')
            for pt in each_object[1]:
                  x=pt[0]
                  point = SubElement(polygon, 'pt')
                  x_point = SubElement(point, 'x')
                  x_point.text = str(x)
                  y=pt[1]
                  y_point = SubElement(point, 'y')
                  y_point.text = str(y)

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = open(self.filename + '.xml', 'w')
        else:
            out_file = open(targetFile, 'w')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class AnnotationReader:

    def __init__(self, filepath):
        # shapes type:
        # [label, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color]
        self.shapes = []
        self.filepath = filepath
        self.parseXML()

    def getShapes(self):
        return self.shapes

    def addShape(self, label, p_type, polygon):
        points=[]
        for pt in polygon.findall('pt'):
            x=int(pt.find('x').text)
            y=int(pt.find('y').text)
            points.append((x,y))
        self.shapes.append((label, p_type, points, None, None))

    def parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        #filename = xmltree.find('filename').text

        for object_iter in xmltree.findall('object'):
            # one polygon per object
            polygon = object_iter.find("polygon")
            p_type_obj=object_iter.find('type')
            p_type=''
            if p_type_obj is None:
                p_type='general'
            else:
                p_type=p_type_obj.text
            label = object_iter.find('name').text
            self.addShape(label, p_type, polygon)
        return True

filepath=r'd:\DATA\Temp\win_20160803_11_28_42_pro.xml'
tempParseReader = AnnotationReader(filepath)
# print tempParseReader.getShapes()
"""
# Test
tmp = AnnotationWriter('temp','test', (10,20,3))
tmp.addBndBox(10,10,20,30,'chair')
tmp.addBndBox(1,1,600,600,'car')
tmp.save()
"""
