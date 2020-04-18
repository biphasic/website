+++
title = "A mixed-signal hardware accelerator for brain machine-interfaces"
date = "2020-01-01"
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["G Haessig", "DC Lesta", "G Lenz", "R Benosman", "P Dudek"]

# Publication type.
# Legend:
# 0 = Uncategorized
# 1 = Conference proceedings
# 2 = Journal
# 3 = Work in progress
# 4 = Technical report
# 5 = Book
# 6 = Book chapter
publication_types = ["1"]

# Publication name and optional abbreviated version.
publication = "Accepted at *International Symposium on Circuits and Systems (ISCAS)*."
publication_short = "Accepted at *ISCAS*."

# Abstract and optional shortened version.
abstract = "Neuromorphic systems provide an alternative to conventional computing hardware, promising low-power operation suitable for sensory-processing and edge computing. In this paper, we present a mixed-signal processing system designed to provide on-sensor classification of signals obtained from multi-electrode array neural recordings. The designed circuits implement a real-time spike sorting algorithm, and operate on signals represented by asynchronous event streams. We combine analog circuits computation primitives (temporal surface generation, distance computation, winner-take-all) to implement a spatio-temporal clustering algorithm, classifying signals acquired by neighbouring electrodes. The prototype chip has been submitted for fabrication in a 180nm CMOS technology. The circuits are designed to fit, alongside signal conditioning and conversion circuits, in the area under the recording electrodes (below 80x80um per electrode). Circuit implementation details and simulation results are presented. The expected neural spike recognition rates of 75% in a single-layer network and 88% in a 2-layer network are comparable with a software implementation, while the system is designed to provide a low-power embedded real-time solution. This work provides a foundation towards the design of a large scale neuromorphic processing system, to be embedded in brain-machine interfaces."

abstract_short = " "

# Featured image thumbnail (optional)
image_preview = ""

# Is this a selected publication? (true/false)
selected = true

# Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter the filename (excluding '.md') of your project file in `content/project/`.
# projects = ["example-external-project"]

# Links (optional).
#url_pdf = ""
#url_preprint = ""
#url_code = "#"
#url_dataset = "#"
#url_project = "#"
#url_slides = "#"
#url_video = "#"
#url_poster = "#"
#url_source = "#"

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
#url_custom = [{name = "Custom Link", url = "http://example.org"}]

# Does the content use math formatting?
math = true

# Does the content use source code highlighting?
highlight = false

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
#[header]
#image = "headers/bubbles-wide.jpg"
#caption = "My caption :smile:"

+++
