from maya.api import OpenMaya as om
import numbers

import logging
_logger = logging.getLogger(__name__)

INT_DATA = [om.MFnNumericData.kShort, om.MFnNumericData.kInt, om.MFnNumericData.kLong, om.MFnNumericData.kByte]
FLOAT_DATA = [om.MFnNumericData.kFloat, om.MFnNumericData.kDouble, om.MFnNumericData.kAddr]


class MPlug(om.MPlug):
    """wrapper of OpenMaya.MPlug to easier the set and get of the values
    Inheritance:
        om.MPlug
    """

    def set(self, inValue):
        """
        Sets the given plug's value to the passed in value.
        Args:
            node (str): node name
            inValue (_Type_): Any value of any data type.
        Raises:
            logger.error: the provide argument has wrong data type for the plug
        """
        plugAttribute = self.attribute()
        apiType = plugAttribute.apiType()
        _logger.debug("Setting {} type {} as {}".format(self.info, plugAttribute.apiTypeStr, inValue))
        # Float Groups - rotate, translate, scale
        if apiType in [om.MFn.kAttribute3Double, om.MFn.kAttribute3Float]:
            if self.isCompound:
                if isinstance(inValue, list) or isinstance(inValue, tuple):
                    for c in xrange(self.numChildren()):
                        MPlug(self.child(c)).set(inValue[c])
                elif type(inValue) in [om.MEulerRotation, om.MVector, om.MPoint]:
                    MPlug(self.child(0)).set(inValue.x)
                    MPlug(self.child(1)).set(inValue.y)
                    MPlug(self.child(2)).set(inValue.z)
                else:
                    _logger.error('{0} :: Passed in value ({1}) is {2}. Needs to be type list.'.format(
                        self.info, inValue, type(inValue)))
        # Distance
        elif apiType in [om.MFn.kDoubleLinearAttribute, om.MFn.kFloatLinearAttribute]:
            if isinstance(inValue, numbers.Number):
                value = om.MDistance(inValue, om.MDistance.kCentimeters)
                self.setMDistance(value)
            else:
                _logger.error('{0} :: Passed in value ({1}) is {2}. Needs to be type number.'.format(
                    self.info, inValue, type(inValue)))
        # Angle
        elif apiType in [om.MFn.kDoubleAngleAttribute, om.MFn.kFloatAngleAttribute]:
            if isinstance(inValue, numbers.Number):
                value = om.MAngle(inValue, om.MAngle.kDegrees)
                self.setMAngle(value)
            else:
                _logger.error('{0} :: Passed in value ({1}) is {2}. Needs to be type number.'.format(
                    self.info, inValue, type(inValue)))
        # Typed - matrix WE DON'T HANDLE THIS CASE YET!!!!!!!!!
        elif apiType == om.MFn.kTypedAttribute:
            pType = om.MFnTypedAttribute(plugAttribute).attrType()
            if pType == om.MFnData.kMatrix:
                if isinstance(inValue, om.MPlug):
                    pass
                else:
                    plugNode = self.node()
                    MFnTrans = om.MFnTransform(plugNode)
                    sourceMatrix = om.MTransformationMatrix(inValue)
                    MFnTrans.set(sourceMatrix)
            # String
            elif pType == om.MFnData.kString:
                value = inValue
                self.setString(value)
        # MATRIX
        elif apiType == om.MFn.kMatrixAttribute:
            if isinstance(inValue, om.MPlug):
                # inValue must be a MPlug!
                sourceValueAsMObject = om.MFnMatrixData(
                    inValue.asMObject()).object()
                self.setMObject(sourceValueAsMObject)
            elif isinstance(inValue, om.MMatrix):
                mtx_data = om.MFnMatrixData()
                mtx_data.create()
                mtx_data.set(inValue)
                self.setMObject(mtx_data.object())
            else:
                _logger.error('Value object is not an MPlug or MMatrix')
        # Numbers
        elif apiType == om.MFn.kNumericAttribute:
            pType = om.MFnNumericAttribute(plugAttribute).numericType()
            if pType == om.MFnNumericData.kBoolean:
                if isinstance(inValue, bool) or isinstance(inValue, numbers.Number):
                    self.setBool(bool(inValue))
                else:
                    _logger.error('{0} :: Passed in value ({1}) is {2}. Needs to be type bool.'.format(
                        self.info, inValue, type(inValue)))
            elif pType in INT_DATA:
                if isinstance(inValue, numbers.Number):
                    self.setInt(inValue)
                else:
                    _logger.error('{0} :: Passed in value ({1}) is {2}. Needs to be type number.'.format(
                        self.info, inValue, type(inValue)))
            elif pType in FLOAT_DATA:
                if isinstance(inValue, numbers.Number):
                    self.setDouble(inValue)
                else:
                    _logger.error('{0} :: Passed in value ({1}) is {2}. Needs to be type number.'.format(
                        self.info, inValue, type(inValue)))
        # Enums TODO: set enum with string
        elif apiType == om.MFn.kEnumAttribute:
            self.setInt(inValue)

    def get(self):
        """
        Gets the value of the given plug.
        Returns:
            variable The value of the passed in node plug.
        """
        pAttribute = self.attribute()
        apiType = pAttribute.apiType()

        # Float Groups - rotate, translate, scale; Compounds
        if apiType in [om.MFn.kAttribute3Double, om.MFn.kAttribute3Float, om.MFn.kCompoundAttribute]:
            result = []
            if self.isCompound():
                for c in xrange(self.numChildren()):
                    result.append(self.get(self.child(c)))
                return result
        # Distance
        elif apiType in [om.MFn.kDoubleLinearAttribute, om.MFn.kFloatLinearAttribute]:
            return self.asMDistance().asCentimeters()
        # Angle
        elif apiType in [om.MFn.kDoubleAngleAttribute, om.MFn.kFloatAngleAttribute]:
            return self.asMAngle().asDegrees()
        # TYPED
        elif apiType == om.MFn.kTypedAttribute:
            pType = om.MFnTypedAttribute(pAttribute).attrType()
            # Matrix
            if pType == om.MFnData.kMatrix:

                return om.MFnMatrixData(self.asMObject()).matrix()
            # String
            elif pType == om.MFnData.kString:
                return self.asString()
        # MATRIX
        elif apiType == om.MFn.kMatrixAttribute:
            return om.MFnMatrixData(self.asMObject()).matrix()
        # NUMBERS
        elif apiType == om.MFn.kNumericAttribute:
            pType = om.MFnNumericAttribute(pAttribute).numericType()
            if pType == om.MFnNumericData.kBoolean:
                return self.asBool()
            elif pType in INT_DATA:
                return self.asInt()
            elif pType in FLOAT_DATA:
                return self.asDouble()
        # Enum
        elif apiType == om.MFn.kEnumAttribute:
            return self.asInt()

    def __getitem__(self, key):
        return MPlug(self.elementByLogicalIndex(key))

    def connectTo(self, destination, force=False):
        """connect current plug to a destination plug

        Args:
            destination (MPlug): plug to recive the connection
            force (bool, optional): if is true it will disconnect any input connections
            to the destination. Defaults to False.

        Raises:
            ValueError: if the destination is not an MPlug
            BaseException: if the destination has input connection and the force flag is False
        """
        if not isinstance(destination, om.MPlug):
            raise ValueError(
                "{} is not an instance of MPlug".format(destination))
        source = destination.source()
        MDGMod = om.MDGModifier()
        if source:
            if source == self:
                _logger.debug("skipping connection already made")
                return
            elif not force:
                raise BaseException("{} is connected to {} try with force argument".format(destination.info,
                                                                                           source.info))
            MDGMod.disconnect(source, destination)
        MDGMod.connect(self, destination)
        MDGMod.doIt()

    def source(self):
        """if has input connection return it, else None

        Returns:
            MPlug, None: the input connections, else None
        """
        mplug = super(MPlug, self).source()
        if mplug.isNull:
            return None
        return MPlug(mplug)

    def destinations(self):
        """return all the plugs where this plug is connected to

        Returns:
            list: all destinations MPlugs
        """
        plugArr = super(MPlug, self).destinations()
        if not plugArr:
            return []
        return [MPlug(a) for a in plugArr if not a.isNull]

    def disconnectSource(self):
        """remove eny input connections that may have
        """
        src_plug = self.source()
        if src_plug.isNull:
            return
        MDGMod = om.MDGModifier()
        MDGMod.disconnect(src_plug, self)
        MDGMod.doIt()

    def disconnectDestinations(self):
        """remove all the connections where this plug is connected to
        """
        dest_plugs = self.destinations()
        MDGMod = om.MDGModifier()
        for dest_plug in dest_plugs:
            if dest_plug.isNull:
                return
            MDGMod.disconnect(self, dest_plug)
        MDGMod.doIt()

    def __getattr__(self, name):
        """get the attribute as MPlug
        Raises:
            AttributeError: if the child attribute is not found 
        Returns:
            mPlug.MPlug
        """
        fn = om.MFnDependencyNode(self.node())
        if not self.isCompound or not fn.hasAttribute(name):
            raise AttributeError("MPlug {} doesn't have property called {}".format(self.info, name))
        child = self.child(fn.attribute(name))
        return MPlug(child)

    def __str__(self):
        """override string operator to return node name
        Returns:
            str: node name
        """
        return self.info

    def __unicode__(self):
        return self.info
