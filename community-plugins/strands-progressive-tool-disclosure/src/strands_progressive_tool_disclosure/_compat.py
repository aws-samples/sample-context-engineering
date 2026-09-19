"""The single point of contact with the SDK's private middleware API.

Every other module of this package reaches the middleware seam only through this module, so an
upstream rename touches one file and the compatibility test has a single target. There is no public
middleware API yet; when one lands, this module is the one adoption point, and the rest of the
package does not change.
"""

from strands._middleware.stages import InvokeModelContext, InvokeModelStage

__all__ = ["InvokeModelContext", "InvokeModelStage"]
