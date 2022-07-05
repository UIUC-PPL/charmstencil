from charmstencil.stencil import Field
from charmstencil.ast import FieldOperationNode

def norm(field, kind):
    node = FieldOperationNode('norm', [field, kind])
    return Field(field.stencil.get_field_name(), 1, field.stencil,
                 graph=node)

