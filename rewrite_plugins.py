import re

with open("src/ml_switcheroo/plugins_legacy.py") as f:
    content = f.read()

classes = re.findall(r'class (\w+):\n\s+"""(.*?)"""\n\n\s+pass', content)

new_content = '"""Plugin Architecture & Legacy Reimplementation Stubs."""\n\nfrom typing import Any, Dict\n\n\nclass LegacyPlugin:\n    """Base class for all legacy reimplementation plugins."""\n\n    def __init__(self, config: Dict[str, Any] = None) -> None:\n        """Initialize the plugin with optional config.\n        \n        Args:\n            config (Dict[str, Any], optional): Configuration dictionary.\n        """\n        self.config = config or {}\n\n    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:\n        """Apply the plugin transformation to the state.\n        \n        Args:\n            state (Dict[str, Any]): The state to transform.\n            \n        Returns:\n            Dict[str, Any]: The transformed state.\n        """\n        state[self.__class__.__name__] = True\n        return state\n\n'

for name, doc in classes:
    new_content += f'\nclass {name}(LegacyPlugin):\n    """{doc}"""\n\n    pass\n'

with open("src/ml_switcheroo/plugins_legacy.py", "w") as f:
    f.write(new_content)

test_content = '"""Tests for legacy plugins."""\n\nfrom ml_switcheroo.plugins_legacy import (\n    LegacyPlugin,\n'
test_content += "    " + ",\n    ".join([name for name, _ in classes]) + "\n)\n\n\n"

test_content += 'def test_base_plugin():\n    """Test the base plugin class."""\n    plugin = LegacyPlugin(config={"a": 1})\n    assert plugin.config == {"a": 1}\n    state = plugin.apply({})\n    assert state["LegacyPlugin"] is True\n\n    plugin2 = LegacyPlugin()\n    assert plugin2.config == {}\n\n'

test_content += (
    'def test_all_plugins():\n    """Test all plugin stubs."""\n    plugins = [\n'
)
test_content += (
    "        " + ",\n        ".join([f"{name}()" for name, _ in classes]) + "\n    ]\n"
)
test_content += "    for plugin in plugins:\n        state = plugin.apply({})\n        assert state[plugin.__class__.__name__] is True\n"

with open("tests/test_plugins_legacy.py", "w") as f:
    f.write(test_content)
