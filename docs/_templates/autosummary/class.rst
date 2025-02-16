{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
        ~{{ objname }}.{{ item }}
    {%- endfor %}
