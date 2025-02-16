{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    .. rubric:: Methods

    {% for item in methods if item != "__init__" %}
    .. automethod:: {{ objname }}.{{ item }}
    {%- endfor %}
