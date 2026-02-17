from werkzeugkoppler.tool_registry import ToolRegistry


def test_map_tool_name_sanitizes():
    assert ToolRegistry._map_tool_name("srv-1", "my.tool") == "srv-1__my_tool"
