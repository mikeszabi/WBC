# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:21:47 2017

@author: SzMike
"""

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


class AnnotationWriter:

    def __init__(self, folder, filename, imagesize, author='Unknown'):
        self.folder = folder
        self.filename = filename
        self.author = author
        self.imagesize = imagesize
        self.shapelist = []
        self.top = None

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
                self.imagesize is None or \
                len(self.shapelist) <= 0:
            return None

        self.top = Element('annotation')

        filename = SubElement(self.top, 'filename')
        filename.text = self.filename

        folder = SubElement(self.top, 'folder')
        folder.text = self.folder

        source = SubElement(self.top, 'source')
        author = SubElement(source, 'submittedBy')
        author.text = self.author

        imagesize = SubElement(self.top, 'imagesize')
        nrows = SubElement(imagesize, 'nrows')
        ncols = SubElement(imagesize, 'ncols')
        nrows.text = str(self.imagesize[0])
        ncols.text = str(self.imagesize[1])

    def addShapes(self, new_shapelist, append=True):
        for new_shapes in new_shapelist:
            assert isinstance(new_shapes[0],str), "Not valid shape label type"
            assert isinstance(new_shapes[1],str), "Not valid shape polygon type"
            #assert isinstance(new_shapes[2],list), "Not valid polygon"
        if append:
            self.shapelist.extend(new_shapelist)
        else:
            self.shapelist=new_shapelist
            
    def appendShapesToXML(self):
        for each_shape in self.shapelist:
            shape_item = SubElement(self.top, 'object')
            name = SubElement(shape_item, 'name')
            name.text = each_shape[0]     
            p_type = SubElement(shape_item, 'type')
            p_type.text = each_shape[1]     
            polygon = SubElement(shape_item, 'polygon')
            for pt in each_shape[2]:
                  x=pt[0]
                  point = SubElement(polygon, 'pt')
                  x_point = SubElement(point, 'x')
                  x_point.text = str(x)
                  y=pt[1]
                  y_point = SubElement(point, 'y')
                  y_point.text = str(y)

    def save(self, targetFile=None):
        self.genXML()
        self.appendShapesToXML()
        out_file = None
        if targetFile is None:
            out_file = open(self.filename + '.xml', 'w')
        else:
            out_file = open(targetFile, 'w')

        prettifyResult = self.prettify(self.top)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class AnnotationReader:

    def __init__(self, filepath):
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

        for shape_iter in xmltree.findall('object'):
            # one polygon per shape
            polygon = shape_iter.find("polygon")
            p_type_obj=shape_iter.find('type')
            p_type=''
            if p_type_obj is None:
                p_type='general'
            else:
                p_type=p_type_obj.text
            label = shape_iter.find('name').text
            self.addShape(label, p_type, polygon)
        return True
"""
filepath=r'd:\DATA\Temp\win_20160803_11_28_42_pro.xml'
tempParseReader = AnnotationReader(filepath)
# print tempParseReader.getShapes()

# Test
tmp = AnnotationWriter('temp','test', (1920,1080))
tmp.addShapes(tempParseReader.shapes)
tmp.save()
"""
