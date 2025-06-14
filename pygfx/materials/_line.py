from ._base import Material
from ..resources import Texture, TextureMap
from ..utils import unpack_bitfield, Color, assert_type
from ..utils.enums import ColorMode, CoordSpace


class LineMaterial(Material):
    """Basic line material.

    Parameters
    ----------
    thickness : float
        The line thickness expressed in logical pixels. Default 2.0.
    thickness_space : str | CoordSpace
        The coordinate space in which the thickness is expressed ('screen', 'world', 'model'). Default 'screen'.
    color : Color
        The uniform color of the line (used depending on the ``color_mode``).
    color_mode : str | ColorMode
        The mode by which the line is coloured. Default 'auto'.
    map : TextureMap | Texture
        The texture map specifying the color for each texture coordinate. Optional.
    dash_pattern : tuple
        The pattern of the dash, e.g. `[2, 3]`. See `dash_pattern` docs for details. Defaults to an empty tuple, i.e. no dashing.
    dash_offset : float
        The offset into the dash phase. Default 0.0.
    loop : bool
        Whether the line's end should be connected. Default False.
    aa : bool
        Whether or not the line is anti-aliased in the shader. Default True.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class <pygfx.Material>`.
    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        thickness="f4",
        dash_offset="f4",
    )

    def __init__(
        self,
        thickness=2.0,
        thickness_space="screen",
        *,
        color=(1, 1, 1, 1),
        color_mode="auto",
        map=None,
        dash_pattern=(),
        dash_offset=0,
        loop=False,
        aa=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.thickness = thickness
        self.thickness_space = thickness_space
        self.color = color
        self.color_mode = color_mode
        self.map = map
        self.dash_pattern = dash_pattern
        self.dash_offset = dash_offset
        self.loop = loop
        self.aa = aa

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, coord=18)
        return {
            "vertex_index": values["index"],
            "segment_coord": (values["coord"] - 100000) / 100000.0,
        }

    def _looks_transparent(self):
        if self.opacity < 1:
            return True
        if self._store.get("color_mode") in ("auto", "uniform"):
            if self.color.a < 1:
                return True

    @property
    def color(self):
        """The uniform color of the line."""
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_full()
        self._resolve_transparent()

    @property
    def aa(self):
        """Whether the line's edges are anti-aliased.

        Aliasing gives prettier results by producing semi-transparent fragments
        at the edges. Lines thinner than one physical pixel are also diminished
        by making them more transparent.

        Note that by default, pygfx uses SSAA to anti-alias the total renderered
        result. Line-based aa results in additional improvement.

        Because semi-transparent fragments are introduced, it may affect how the
        line blends with other (semi-transparent) objects.
        """
        return self._store.aa

    @aa.setter
    def aa(self, aa):
        self._store.aa = bool(aa)

    @property
    def color_mode(self):
        """The way that color is applied to the line.

        See :obj:`pygfx.utils.enums.ColorMode`:
        """
        return self._store.color_mode

    @color_mode.setter
    def color_mode(self, value):
        value = value or "auto"
        if value not in ColorMode:
            raise ValueError(
                f"LineMaterial.color_mode must be a string in {ColorMode}, not {value!r}"
            )
        self._store.color_mode = value
        self._resolve_transparent()

    @property
    def vertex_colors(self):
        return self.color_mode == ColorMode.vertex

    @vertex_colors.setter
    def vertex_colors(self, value):
        raise DeprecationWarning(
            "vertex_colors is deprecated, use ``color_mode='vertex'``"
        )

    @property
    def thickness(self):
        """The line thickness.

        The interpretation depends on `thickness_space`. By default it is in logical
        pixels, but it can also be in world or model coordinates.
        """
        return float(self.uniform_buffer.data["thickness"])

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = max(0.0, float(thickness))
        self.uniform_buffer.update_full()

    @property
    def thickness_space(self):
        """The coordinate space in which the thickness (and dash_pattern) are expressed.

        See :obj:`pygfx.utils.enums.CoordSpace`:
        """
        return self._store.thickness_space

    @thickness_space.setter
    def thickness_space(self, value):
        value = value or "screen"
        if value not in CoordSpace:
            raise ValueError(
                f"LineMaterial.thickness_space must be a string in {CoordSpace}, not {value!r}"
            )
        self._store.thickness_space = value

    @property
    def map(self):
        """The texture map specifying the color for each texture coordinate.

        Can be None. The dimensionality of the map can be 1D, 2D or 3D, but
        should match the number of columns in the geometry's texcoords.
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert_type("map", map, None, Texture, TextureMap)
        if isinstance(map, Texture):
            map = TextureMap(map)
        self._store.map = map

    @property
    def dash_pattern(self):
        """The dash pattern.

        A sequence of floats describing the length of strokes and gaps. The
        length of the sequence must be an even number. Setting to None or the
        empty tuple means no dashing.

        For example, (5, 2, 1, 2) describes a a stroke of 5 units, a gap of 2,
        then a short stroke of 1, and another gap of 2. Units are relative to
        the line thickness (and therefore `thickness_space` also applies to  the
        `dash_pattern`).
        """
        return self._store.dash_pattern

    @dash_pattern.setter
    def dash_pattern(self, value):
        if value is None:
            value = ()
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "Line dash_pattern must be a sequence of floats, not '{value}'"
            )
        if len(value) % 2:
            raise ValueError("Line dash_pattern must have an even number of elements.")
        self._store.dash_pattern = tuple(max(0.0, float(v)) for v in value)

    @property
    def dash_offset(self):
        """The offset into the dash cycle to start drawing at, i.e. the phase."""
        return float(self.uniform_buffer.data["dash_offset"])

    @dash_offset.setter
    def dash_offset(self, value):
        self.uniform_buffer.data["dash_offset"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def loop(self) -> bool:
        """Whether the line's ends should be connected.

        If set to True, the end of the line is connected to its beginning, in
        such a way there is no overlap (which would otherwise be visible for
        semi-transparent lines). When the line consists of multiple pieces
        separated by nan-positions, each line-piece is considered a loop.
        """
        return self._store.loop

    @loop.setter
    def loop(self, loop: bool):
        self._store.loop = bool(loop)


class LineDebugMaterial(LineMaterial):
    """Line debug material.

    A material that renders the triangles that the line is made up off.
    """

    pass


class LineSegmentMaterial(LineMaterial):
    """Line segment material.

    A material that renders line segments between each two subsequent points.
    """


class LineInfiniteSegmentMaterial(LineSegmentMaterial):
    """Infinite line segment material.

    A material that renders infenitely long line segments between each two
    subsequent points. The end-points of each segment are displaced (along the
    vector defined by the two points) such that the points are at the edge of
    the viewport. Other than that, dashing, vertex colors, etc. should work as
    expected (interpolating between the points that are now on the viewport edge).

    Parameters
    ----------
    start_is_infinite : bool
        Whether start of each segment is made infinitely long. Default True.
    end_is_infinite : bool
        Whether end of each segment is made infinitely long. Default True.
    """

    def __init__(self, start_is_infinite=True, end_is_infinite=True, **kwargs):
        super().__init__(**kwargs)
        self.start_is_infinite = start_is_infinite
        self.end_is_infinite = end_is_infinite

    @property
    def start_is_infinite(self):
        """Whether start of each segment is made infinitely long."""
        return self._store.start_is_infinite

    @start_is_infinite.setter
    def start_is_infinite(self, value):
        self._store.start_is_infinite = bool(value)

    @property
    def end_is_infinite(self):
        """Whether end of each segment is made infinitely long."""
        return self._store.end_is_infinite

    @end_is_infinite.setter
    def end_is_infinite(self, value):
        self._store.end_is_infinite = bool(value)


class LineArrowMaterial(LineSegmentMaterial):
    """Arrow (vector) line material.

    A material that renders line segments that look like little arrows.
    """


class LineThinMaterial(LineMaterial):
    """Thin line material.

    A simple line, drawn with line_strip primitives that has a thickness
    of one physical pixel. Thickness, dashing, and aa are ignored.

    While you typically don't want to use this in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """


class LineThinSegmentMaterial(LineMaterial):
    """Thin line segment material.

    Simple line segments, drawn with line primitives that has a thickness
    of one physical pixel. Thickness, dashing, and aa are ignored.

    While you typically don't want to use this in your application (its
    width is inconsistent and looks *very* thin on HiDPI monitors), it can be
    useful for debugging as it is more performant than other line materials.

    """
