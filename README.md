# Point-in-Time WCC Model

*Real-time component retrieval in constant time per member, for temporal graphs in Neo4j*

---

## The Problem: Future Leakage in Graph-Based Feature Engineering

Weakly Connected Components (WCC) are powerful features for fraud detection ML. When entities share identifiers across events — a phone number reused across accounts, a device in multiple transactions — WCC captures hidden structure that individual records cannot. Component size, growth velocity, and diameter become strong signals.

But **WCC is not a point-in-time operation.**

Running WCC on the current graph produces the component as it exists *today* — not as it existed when a given event occurred. Using this as a feature for historical records leaks future information into training data.

This is **future leakage**, and it invalidates the model.

Example: event `e₁` at time `t₁` belongs to a component of 3 events. By `t₂`, that component has merged with another and contains 50. Assigning `component_size = 50` to `e₁` leaks knowledge of 47 events that hadn't happened yet. The model exploits information unavailable at inference time — metrics inflate during training and collapse in production.

The Point-in-Time WCC Model embeds the full history of component evolution into the graph, enabling:

- **Component retrieval as of any event's own timestamp** — for leak-free ML training data
- **Component retrieval as of any arbitrary date** — for historical analysis
- **Component retrieval as of now** — for real-time inference
- **Single-digit millisecond query performance** — via pre-materialized traversal chains, no WCC recomputation required


## Architecture: Training vs. Inference

The model serves two paths sharing the same data model on different infrastructure.

### Training Path

Raw data is loaded into an **analytics Neo4j instance** with the Graph Data Science (GDS) library. The batch pipeline builds the temporal component structure. Point-in-time queries extract features for each event **as of its own timestamp**, producing a leak-free training dataset.

### Inference Path

New events arrive at an **HA Neo4j transactional cluster**. Each is ingested via a single Cypher transaction that atomically updates the component structure. Features are extracted immediately and passed to the trained model for real-time scoring.

Same data model, same queries, both paths.


## Foundation: The Event–Thing Bipartite Graph

The model builds on a domain-agnostic bipartite structure:

```
(:Event {event_id, timestamp})-[:WITH]->(:Thing {thing_id})
```

An **Event** is any timestamped occurrence — a transaction, a login, a claim. A **Thing** is any identifier observed during that event — a phone number, a device fingerprint, an account.

Two events sharing a Thing are implicitly connected. The union of these connections defines the graph's weakly connected components: clusters of events linked by shared identifiers over time.

Thing nodes carry domain-specific labels as subtypes (`:Phone`, `:Device`, `:Account`, etc.). The model is agnostic to these — it operates on the Event–Thing topology alone.


## The Temporal Component Structure

Three relationship types and one auxiliary label overlay the Event–Thing foundation.

### Node Labels

| Label | Description |
|---|---|
| `Event` | Carries `event_id` and `timestamp`. |
| `Thing` | Carries `thing_id`. Domain-specific subtypes via labels. |
| `ComponentNode` | Added to every Event once processed into the component structure. |

### Relationship Types

| Relationship | Direction | Description |
|---|---|---|
| `WITH` | `Event → Thing` | Base bipartite link. |
| `COMPONENT_PARENT` | `ComponentNode → ComponentNode` | Points forward in time from old head to new. Forest: out-degree ≤ 1, in-degree ≥ 0. |
| `DFS_NEXT` | `ComponentNode → ComponentNode` | Pre-materialized traversal order. Linked list from head through all members. |
| `LAST_DFS_NODE_IN_COMP` | `ComponentNode → ComponentNode` | Head → last node in its DFS chain. Enables bounded traversal. |

### Structural Invariants

`COMPONENT_PARENT` forms a **chronologically oriented forest**:
- **Out-degree 0 or 1.** A head either has no successor (current head) or one (superseded by a later event).
- **In-degree ≥ 0.** Merges create multiple incoming edges on the new head.
- **Roots are current heads** — no outgoing `COMPONENT_PARENT`.

Each head's `DFS_NEXT` chain is a **linked list** from head to the node referenced by `LAST_DFS_NODE_IN_COMP`, containing every member of that component at that point in time.

### How Components Evolve

When a new event `e` arrives:

**Case 1: No match.** All Things are new. `e` becomes a standalone component with `LAST_DFS_NODE_IN_COMP` pointing to itself.

**Case 2: Single match.** Things overlap with one existing component (head `h`, tail `z`):
- `h -[:COMPONENT_PARENT]-> e`
- `e -[:DFS_NEXT]-> h`
- `e -[:LAST_DFS_NODE_IN_COMP]-> z`

**Case 3: Merge.** Things span N ≥ 2 components with heads `h₁, h₂, ..., hₙ` (ordered by `event_id`):
- All old heads get `COMPONENT_PARENT → e`
- DFS chains are **concatenated**: tail(h₁) `→ DFS_NEXT →` h₂, etc.
- `e -[:DFS_NEXT]-> h₁`
- `e -[:LAST_DFS_NODE_IN_COMP]-> tail(hₙ)`

Old heads are never deleted — only a `COMPONENT_PARENT` edge is added. Every historical state is preserved.


## Querying the Structure

All three queries walk the pre-materialized `DFS_NEXT` linked list. No WCC recomputation, no graph-wide traversal.

### Component as of Event Time

Retrieve the component as it existed when a given event occurred:

```cypher
// Component as of event's own timestamp
MATCH path = (e:Event {event_id: $event_id})
  (()-[:DFS_NEXT]->(evs))*
  (last)<-[:LAST_DFS_NODE_IN_COMP]-(e)
UNWIND [e] + evs AS ev
RETURN ev
```

Walks the DFS chain from the event to the `LAST_DFS_NODE_IN_COMP` boundary. Returns every event in the component at the moment `$event_id` was recorded. This is the query for **leak-free ML feature engineering**.

### Component as of Now (Latest State)

Retrieve the current component for a given event:

```cypher
// Component as of today
MATCH fast_fw_to_future = (e:Event {event_id: $event_id})
  -[:COMPONENT_PARENT]->*
  (latest_future_comp:Event
    WHERE NOT EXISTS {
      (latest_future_comp)-[:COMPONENT_PARENT]->()
    }
  ),
  comp = (latest_future_comp)-[:DFS_NEXT]->*(last),
  (last)<-[:LAST_DFS_NODE_IN_COMP]-(latest_future_comp)
UNWIND nodes(comp) AS ev
RETURN ev
```

Walks `COMPONENT_PARENT` to the current root (no outgoing edge), then walks its `DFS_NEXT` chain. The fast-forward is linear since out-degree is at most 1.

### Component as of Arbitrary Date

Retrieve the component at a specific point in time:

```cypher
// Component as of $asOfDate
MATCH fast_fw_to_future = (e:Event {event_id: $event_id}
    WHERE e.timestamp <= $asOfDate)
  (()-[:COMPONENT_PARENT]->(future WHERE future.timestamp <= $asOfDate))*
  (latest_future_comp:Event
    WHERE NOT EXISTS {
      (latest_future_comp)-[:COMPONENT_PARENT]->(x
        WHERE x.timestamp <= $asOfDate)
    }
  ),
  comp = (latest_future_comp)-[:DFS_NEXT]->*(last),
  (last)<-[:LAST_DFS_NODE_IN_COMP]-(latest_future_comp)
UNWIND nodes(comp) AS ev
RETURN ev
```

Filters the `COMPONENT_PARENT` walk by timestamp, stopping at the latest head on or before `$asOfDate`.

### Performance

Pre-materialized DFS chains mean component retrieval is a linked-list walk — **constant time per member**. No WCC recomputation, no GDS call, no traversal through Things. Typical queries complete in **single-digit milliseconds** even for components with thousands of members.


## Building the Structure: Batch Pipeline

The batch pipeline builds the structure from an existing Event–Thing graph using the Neo4j Graph Data Science (GDS) library in four steps.

### Prerequisites

```cypher
CREATE CONSTRAINT event_id_unique IF NOT EXISTS
  FOR (e:Event) REQUIRE (e.event_id) IS UNIQUE;
CREATE CONSTRAINT thing_id_unique IF NOT EXISTS
  FOR (t:Thing) REQUIRE (t.thing_id) IS UNIQUE;
CREATE INDEX event_timestamp IF NOT EXISTS
  FOR (e:Event) ON (e.timestamp);
```

### Step 1: Build SEQUENTIALLY_RELATED

For each Thing, time-order its Events and chain consecutive pairs. This turns implicit bipartite connectivity into explicit Event-to-Event edges.

```cypher
// Project the bipartite graph
MATCH (source:Event)
OPTIONAL MATCH (source)-[:WITH]->(target)
RETURN gds.graph.project('event_thing_graph', source, target, {});
```

```cypher
// Create SEQUENTIALLY_RELATED via WCC-batched processing
CALL gds.wcc.stream('event_thing_graph')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS thing, componentId AS community
WHERE thing:Thing
WITH community, collect(thing) AS things
CALL (things) {
  UNWIND things AS thing
  CALL (thing) {
    MATCH (e:Event)-[:WITH]->(thing)
    WITH DISTINCT e ORDER BY e.timestamp
    WITH collect(e) AS events
    UNWIND range(0, size(events)-2) AS ix
    WITH events[ix] AS source, events[ix+1] AS target
    MERGE (source)-[:SEQUENTIALLY_RELATED]->(target)
  }
} IN 8 CONCURRENT TRANSACTIONS OF 100 ROWS
```

WCC batches the processing by component for efficiency — without this pre-pass, SEQUENTIALLY_RELATED creation would iterate over all Things globally instead of scoping each to its connected subgraph. This technique is inspired by [Maxime Guéry's approach to avoiding Cypher query crashes via WCC-scoped batching](https://neo4j.com/blog/developer/wcc-to-avoid-cypher-query-crashing/), as cited in [Pierre Halftermeyer's article on temporal graph modeling for fraud detection](https://neo4j.com/blog/developer/mastering-fraud-detection-temporal-graph/).

After this step, the SEQUENTIALLY_RELATED graph encodes the same WCC structure as the original bipartite graph.

> **Note:** `SEQUENTIALLY_RELATED` is intermediate scaffolding that simplifies the COMPONENT_PARENT build by making event-to-event connectivity explicit. Two approaches are possible:
> 1. **Persist it** (shown above) — useful if incremental batches are planned.
> 2. **Project it in memory** — compute the same consecutive-event-per-Thing pairs inline and project them directly into GDS without persisting any relationships:
>
> ```cypher
> MATCH (thing:Thing)
> CALL (thing) {
>   MATCH (e:Event)-[:WITH]->(thing)
>   WITH DISTINCT e
>   WITH collect(e) AS events
>   WITH CASE size(events)
>     WHEN 1 THEN [events[0], null]
>     ELSE events END AS events
>   UNWIND range(0, size(events)-2) AS ix
>   RETURN events[ix] AS source, events[ix+1] AS target
> }
> RETURN gds.graph.project(
>   'seq_rel_event_graph', source, target, {}
> );
> ```

### Step 2: Build COMPONENT_PARENT Forest

Project the SEQUENTIALLY_RELATED graph and process events chronologically within each WCC to build the union-find forest:

```cypher
// Project the event-only graph (unless option 2 above already produced it)
MATCH (source:Event)
OPTIONAL MATCH (source)-[:SEQUENTIALLY_RELATED]->(target)
RETURN gds.graph.project('seq_rel_event_graph', source, target, {});
```

```cypher
// Build the COMPONENT_PARENT forest
CALL gds.wcc.stream('seq_rel_event_graph')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS event, componentId
WITH componentId, collect(event) AS events
ORDER BY rand()  // distribute components across concurrent transactions
CALL (events) {
  UNWIND events AS e
  WITH e WHERE NOT e:ComponentNode
  ORDER BY e.timestamp ASC
  CALL (e) {
    SET e:ComponentNode
    WITH e
    MATCH (x:ComponentNode)-[:SEQUENTIALLY_RELATED]->(e)
    MATCH (x)-[:COMPONENT_PARENT]->*(cc
      WHERE NOT EXISTS {(cc)-[:COMPONENT_PARENT]->()})
    MERGE (cc)-[:COMPONENT_PARENT]->(e)
  }
} IN 8 CONCURRENT TRANSACTIONS OF 100 ROWS
```

Replays the online algorithm in batch: events are processed in timestamp order, each finding its neighbors' current heads. `ORDER BY rand()` distributes independent components across concurrent transactions.

### Step 3: Build DFS_NEXT Chains

Project the COMPONENT_PARENT forest (reversed) and run GDS DFS from each root:

```cypher
// Project the component forest (reversed direction for DFS from roots)
MATCH (source:Event)
OPTIONAL MATCH (source)<-[:COMPONENT_PARENT]-(target)
RETURN gds.graph.project('component_forest', source, target, {});
```

```cypher
// Create DFS_NEXT from root component heads
MATCH (source:ComponentNode)
WHERE NOT EXISTS {(source)-[:COMPONENT_PARENT]->()}
  AND EXISTS {(source)<-[:COMPONENT_PARENT]-()}
CALL (source) {
  CALL gds.dfs.stream('component_forest', {
    sourceNode: source
  })
  YIELD path
  WITH relationships(path) AS rels
  UNWIND rels AS rel
  WITH startNode(rel) AS n1, endNode(rel) AS n2
  MERGE (n1)-[:DFS_NEXT]->(n2)
} IN 8 CONCURRENT TRANSACTIONS OF 50 ROWS
```

### Step 4: Build LAST_DFS_NODE_IN_COMP Markers

Create boundary markers for bounded DFS chain traversal:

```cypher
CYPHER 25
CALL () {
    // Case 1: ComponentNode with a DFS_NEXT successor — find the last node
    // in its own sub-tree (nodes reachable via COMPONENT_PARENT from it,
    // but not from its DFS_NEXT successor)
    MATCH (c1:ComponentNode)-[:DFS_NEXT]->(c2:ComponentNode)
    MATCH (c1)(()-[:COMPONENT_PARENT]->(ps)
      WHERE NOT EXISTS {(c2)-[:COMPONENT_PARENT]->*(ps)}
    )*
    UNWIND ps AS p
    RETURN c1, p

  UNION

    // Case 2: ComponentNode with no DFS_NEXT successor — it is a leaf,
    // walk its full COMPONENT_PARENT sub-tree
    MATCH (c1:ComponentNode WHERE NOT EXISTS {(c1)-[:DFS_NEXT]->()})
    MATCH (c1)(()-[:COMPONENT_PARENT]->(ps))*
    UNWIND ps AS p
    RETURN c1, p

  UNION

    // Case 3: ComponentNode with no incoming COMPONENT_PARENT —
    // standalone single-event component, points to itself
    MATCH (c1:ComponentNode WHERE NOT EXISTS {()-[:COMPONENT_PARENT]->(c1)})
    RETURN c1, c1 AS p

}
CALL (c1, p) {
  MERGE (p)-[:LAST_DFS_NODE_IN_COMP]->(c1)
} IN TRANSACTIONS OF 100 ROWS
```


## Building the Structure: Online (Single Event Ingestion)

A single Cypher query atomically updates the component structure on each new event. No GDS required.

```cypher
CYPHER 25
LET event = {
  event_id: $event_id,
  things: $things
}
WITH event, collect {
  UNWIND event.things AS th
  MATCH (t:Thing {thing_id: th.thing_id})<-[:WITH]-(c:ComponentNode)
  RETURN c
} AS matched_comps
CALL (event, matched_comps) {
  // Case 1: No existing components matched — create standalone component
  WHEN size(matched_comps) = 0 THEN {
    CREATE (e:Event {event_id: event.event_id})
    SET e.timestamp = datetime(),
        e:ComponentNode
    UNWIND event.things AS th
    MERGE (t:Thing {thing_id: th.thing_id})
    ON CREATE SET t:$(th.labels)
    MERGE (e)-[:WITH]->(t)
    MERGE (e)-[:LAST_DFS_NODE_IN_COMP]->(e)
    RETURN e
  }
  // Case 2+3: One or more existing components matched — extend or merge
  ELSE {
    CREATE (e:Event {event_id: event.event_id})
    SET e.timestamp = datetime(),
        e:ComponentNode
    UNWIND event.things AS th
    MERGE (t:Thing {thing_id: th.thing_id})
    ON CREATE SET t:$(th.labels)
    MERGE (e)-[:WITH]->(t)
    // Walk each Thing's ComponentNode up to its current root
    CALL (t) {
      MATCH (t)<-[:WITH]-(ev:ComponentNode)
      LIMIT 1  // any entry point reaches the same root via COMPONENT_PARENT
      MATCH (ev)-[:COMPONENT_PARENT]->*(parent
        WHERE NOT EXISTS {(parent)-[:COMPONENT_PARENT]->()}
      )
      RETURN parent
    }
    WITH DISTINCT event, e, parent
    ORDER BY parent.event_id  // deterministic concatenation order
    WITH event, e, collect(parent) AS sub_comps
    // Stitch DFS chains of sub-components together
    CALL (sub_comps) {
      UNWIND range(0, size(sub_comps)-2) AS ix
      WITH sub_comps[ix] AS comp1, sub_comps[ix+1] AS comp2
      MATCH (comp1)-[:LAST_DFS_NODE_IN_COMP]->(last)
      MERGE (last)-[:DFS_NEXT]->(comp2)
    }
    // Point all old heads to the new event
    CALL (e, sub_comps) {
      UNWIND sub_comps AS comp
      MERGE (comp)-[:COMPONENT_PARENT]->(e)
    }
    // Prepend new event and set new tail marker
    WITH event, e, sub_comps,
         sub_comps[0] AS first_comp,
         sub_comps[-1] AS last_comp
    MATCH (last_comp)-[:LAST_DFS_NODE_IN_COMP]->(last)
    MERGE (e)-[:DFS_NEXT]->(first_comp)
    MERGE (e)-[:LAST_DFS_NODE_IN_COMP]->(last)
    RETURN e
  }
}
RETURN e.event_id AS created_event
```

Handles all three cases atomically:
- **No match:** Self-referencing single-event component.
- **Single match:** Old head gets `COMPONENT_PARENT` to new event; new event gets `DFS_NEXT` to old head and inherits `LAST_DFS_NODE_IN_COMP`.
- **Merge:** All old heads get `COMPONENT_PARENT` to new event; DFS chains concatenated; new event prepended.

> **Important:** Online ingestion must be single-threaded to maintain the forest invariant (COMPONENT_PARENT out-degree ≤ 1).


## Feature Engineering

Features are computed per event using its point-in-time component and stored on the Event node.

### Example Features

**Component size** — number of events in the component. Rapid growth may signal coordinated activity.

**Component diameter** — longest shortest path in the bipartite subgraph, normalized for bipartite hops. Measures how spread out the shared-identifier network is.

**Component velocity** — events per second across the component's time span. High velocity may indicate automation.

### Feature Computation Query

```cypher
CYPHER 25
MATCH (source:Event)
CALL (source) {

    // Retrieve point-in-time component
    WITH source, COLLECT {
        MATCH (source)(()-[:DFS_NEXT]->(evs))*(last)
              <-[:LAST_DFS_NODE_IN_COMP]-(source)
        UNWIND [source] + evs AS ev
        RETURN ev
    } AS cc_elements
    RETURN source, cc_elements, toString(source.event_id) AS graph_name
    NEXT

    // Compute features using GDS on the component subgraph
    CALL (source, cc_elements, graph_name) {
        UNWIND cc_elements AS element
        MATCH (element)
        OPTIONAL MATCH (element)-[:WITH]->(thing)
        WITH gds.graph.project(
            graph_name,
            element,
            thing,
            {},
            {undirectedRelationshipTypes: ['*']}
        ) AS graph
        CALL (cc_elements, graph_name, graph) {
            WHEN graph IS NOT NULL AND graph.relationshipCount > 0 THEN {
                UNWIND cc_elements AS element
                CALL gds.allShortestPaths.delta.stream(graph_name, {
                    sourceNode: element
                })
                YIELD path
                WITH length(path) AS length
                RETURN max(length) / 2 AS diameter
            }
            ELSE {
                RETURN INFINITY AS diameter
            }
        }
        CALL gds.graph.drop(graph_name, false)
        YIELD graphName
        WITH source, size(cc_elements) AS nb_elements, diameter
        SET source.component_size = nb_elements,
            source.component_diameter = diameter
        // Compute velocity
        WITH source, cc_elements
        UNWIND cc_elements AS element
        WITH source, element ORDER BY element.timestamp ASC
        WITH source, collect(element) AS ordered_elements
        WITH
          source,
          duration.inSeconds(
            ordered_elements[0].timestamp,
            ordered_elements[-1].timestamp
          ).seconds AS time_span,
          size(ordered_elements) AS nb_events
        SET source.component_velocity =
          CASE WHEN time_span > 0
          THEN toFloat(nb_events) / time_span
          ELSE 0.0
          END
    }
} IN 6 CONCURRENT TRANSACTIONS OF 100 ROWS
```

Projects each component's bipartite subgraph into GDS on the fly, computes diameter via delta-stepping shortest paths, and derives velocity from the time span. Every feature is temporally accurate — computed against the component as it existed when that event occurred.


## Summary

The Point-in-Time WCC Model solves temporal leakage in WCC-based feature engineering by storing the full evolution of component membership in the graph via three overlay relationships: `COMPONENT_PARENT` (versioning), `DFS_NEXT` (traversal), and `LAST_DFS_NODE_IN_COMP` (bounded walks).

It supports batch construction (GDS) for training and single-event online ingestion (pure Cypher) for inference. Component retrieval at any point in time is a linked-list walk with single-digit millisecond latency.

The Event–Thing bipartite foundation is domain-agnostic. Any use case with timestamped events sharing identifiers — fraud detection, identity resolution, network analysis, supply chain monitoring — plugs in with minimal adaptation.
