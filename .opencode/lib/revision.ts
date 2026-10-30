/**
 * Revision of the plugin's retrieval behaviour. The runner records it in every arm-A evidence, so a result
 * always says which behaviour produced it.
 *
 *  1  scoped excerpt notice, narrow reads (pattern / line_range / max_chunks), retrieval exchanges DELETED from
 *     the model-visible history after the turn (campaign v5, R1). Two defects: references stopped resolving 300 s
 *     after storage (no lease renewal), and deleting the call made the model re-run the retrieval every turn.
 *  2  + lease renewal on every request plus a periodic heartbeat, and retrieval results CLEARED instead of
 *     deleted (OpenCode's own "[Old tool result content cleared]" marker; the call stays visible) (campaign v6).
 */
export const PLUGIN_REVISION = 2;
