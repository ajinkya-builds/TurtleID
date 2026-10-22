# Naming a seized turtle

A plain-language briefing on the problem, the published record, and what the TurtleID app currently does. Compiled 12 October 2026 from the conversation with Prashant Jijaji (8 October 2026), this repository, and published seizure studies.

The live prototype is [turtleidwct.streamlit.app](https://turtleidwct.streamlit.app/). In the code it is called Chelonid-ID.

---

## The job, in the words of the person who started it

Prashant’s idea was unfinished. It was meant for people in the forest department who confiscate turtles and tortoises in trade. They need a species name, where that species actually lives, and enough care guidance to keep the animals alive until they can go back to the right place. > **Ajinkya:** A reference module that store details of all types of species of turtles and vends it out when there's is a successfull or potential matches.

He was clear about the boundary. This is identification of animals already in hand after a seizure. It is a demand-based tool: whenever a team comes across a consignment, they should be able to name what they are holding. It is not a survey of wild turtles on the land, and it is not a population census. > **Ajinkya:** A web based too for someone to come and run the analysis. Open question if it should be accessible to everyone considering risks for the traffickers to abuse it.

The intended users are the State Tiger Strike Force. Their job is to enforce the Wild Life (Protection) Act, 1972. In practice that means seizures of trafficked animals. Turtles are a large part of that work in Madhya Pradesh, which sits on the rail routes between the rivers where animals are caught and the markets they are heading toward.

He currently answers identification requests himself, about every other week, for batches of 20 to 300 animals. He looks at photographs. The marks that separate species are blotches on the belly shell and the back shell, and sometimes stripes or dots on the neck and face. India has about 28 to 30 freshwater turtles and tortoises. He wanted a reference built from published books and papers, then improved with photographs that users upload. He was thinking of pattern matching, and he pointed to a 2026 paper that uses that family of methods. That paper is discussed at the end. It studies a different problem. >**Ajinkya:**  Lets dig up literature and books and papers that allow us to train a image analysis model with the help of AI to be able to do this processing - We need absolute highest accuracy to be able to do this

How often seized turtles die after being released in the wrong place is, in his words, unknown and not compiled. The published record supports that. India has good counts of what was seized. It has very little follow-up on what happened to the animals afterwards. > **Ajinkya:** Question for the users if we want to further track it? Perhaps do analysis on the area the turtle is being released to to ensure suitability.

---



## What the problem actually is

Three decisions depend on getting the species right, and they all happen in the hours after a seizure, often without a turtle specialist in the room.

**The case file.** Under the Wild Life (Protection) Act, the schedule an animal sits on changes the offence. After the 2022 amendment, the field reference used by this app treats almost every native Indian freshwater turtle and tortoise as Schedule I, with three exceptions on Schedule II: the Indian black turtle (*Melanochelys trijuga*), the brown roofed turtle (*Pangshura smithii*), and the Assam leaf turtle (*Cyclemys gemeli*). Two animals that look alike can sit on different schedules. The Indian roofed turtle and the brown roofed turtle are separated largely by the colour of the belly shell. One is Schedule I and the other is Schedule II. Writing the wrong name on a seizure memo is a legal error, not a clerical one.

Older studies describe the law as it stood before 2022, when many species sat on lower schedules or on none. Those papers are still the best counts of the trade. Their legal labels are out of date.

**Where the animal can go.** A turtle taken from the Ganga and released into a dam in another river basin is not “back in the wild.” Species differ in the rivers, forests, and climates they can live in. Softshell turtles taken for meat, star tortoises taken for the pet trade, and river turtles such as the red-crowned roofed turtle do not share a home. Releasing a mixed crate into the nearest water body treats them as if they do.

**Whether it survives the week.** Animals arrive dehydrated, packed in bags, and often already sick. They then need the right water, the right temperature, and the right food until someone can move them. That is the husbandry part of the original idea. Identification is what makes the other two decisions possible. Without a name, neither the charge nor the release site can be chosen properly.

The practical difficulty is that the species are separated by small marks, the consignments are large, and the officer on the platform is not a turtle taxonomist. Prashant is currently that taxonomist, by phone, every other week.

---



## What the published record shows

These numbers are seizures that were reported. They are a floor under the trade, not the trade itself. In a large share of cases the species was never written down, so the totals mix named animals with “turtle.”

### The trade is large, and the name is often missing

Mendiratta and colleagues, in *Biological Conservation* (2017), compiled 223 seizure reports from 2011 to 2015. They found at least 58,442 live tortoises and freshwater turtles, from at least 15 of India’s then 28 species, including 10 species the IUCN lists as threatened. Nearly 90 percent of seizures were commercial. Animals moved by road, rail, and air inside India, and onward to Bangladesh, Thailand, and other East and Southeast Asian countries. The three species that dominated were the Indian star tortoise, the spotted pond turtle, and the Indian flapshell turtle. The authors’ own conclusion was that enforcement agencies need the capacity to handle these seizures, not only to intercept them.

TRAFFIC India, with the Turtle Survival Alliance and WWF-India, then counted a longer window. *Tortoises and Freshwater Turtles in illegal trade in India* (Badola, Choudhary and Chhabra, 2019) reported a minimum of 111,312 animals from September 2009 to September 2019: more than 11,000 a year, on the order of 200 a week. Fourteen Indian species appeared. In 51.5 percent of the seizure cases, the species was not reported. TRAFFIC’s reading was direct: officers often could not identify the animals, and the Act punishes different species differently, so a missing name weakens the case. That study is why TRAFFIC, TSA-India, and WWF-India issued pocket identification cards for 28 species in 2019 and updated them in 2023 for the amended Act. The cards are still the reference Indian enforcement agencies are expected to carry. They are paper. They do not look at a photograph.

A later India-wide network study, Sengottuvel and colleagues in *Oryx* (2024), is cited by subsequent work as 268 cases and about 74,000 animals from 2013 to 2019, and as showing that softshell turtles (mostly meat) and hardshell turtles and tortoises (mostly pets) move on different routes. Uttar Pradesh supplies much of the softshell trade.

West Bengal, studied by Mandal, Halder Mallick, and Bhattacharya from Forest Department records and news reports for 2017–2023 (preprint, 2024), accounted for 133 confiscations and 33,317 animals of nine species. About 94 percent of the identified individuals were Indian flapshell turtles, consistent with a meat trade, much of it moving toward Bangladesh. North 24 Parganas, on that border, held 57 percent of incidents. One red-eared slider, an American pet turtle, was among the animals seized. The authors close on the point Prashant is worried about: live turtles are often released into the local water body. They name the risks as disease, mixing of populations that should stay separate, and harm to whatever already lives there. They ask for housing, a health check, and a check on where the animals came from before any release.

### Madhya Pradesh is where many of these animals are intercepted

The Indian Express, reporting enforcement records, said that between 2020 and 2026 the State Tiger Strike Force and the Government Railway Police recovered 654 protected animals from passenger trains crossing Madhya Pradesh. Of those, 647 were turtles. Officers described freshwater turtles as the animals they meet most often in trafficking cases. The state is a transit corridor. Animals are taken from the Ganga and its tributaries in Uttar Pradesh, hidden in coaches, and moved toward western and southern markets. The routes named in that reporting are Patna–Lucknow–Indore, Kanpur–Bhopal–Mumbai, and Lucknow–Itarsi–Bengaluru.

Recent cases sit in the same size range Prashant described:

- September 2023, Yeshwantpur Express: 282 Indian tent turtles, 280 alive, intercepted by the Strike Force with the Directorate of Revenue Intelligence, in transit from Lucknow toward Chennai. A trial court later sentenced four men to three years. The surviving animals were released into the Tawa reservoir, beside Satpura Tiger Reserve. Tent turtles do live in Madhya Pradesh, including in the Narmada system that the Tawa feeds. The animals in that case were described as taken from the Ganga. The Ganga and Narmada animals are different populations, treated in the reference database here as different subspecies. A correct species name still leaves a second question: which river they came from.
- February 2026, Patna–Indore Express at Bhopal: 311 turtles, reported as crowned river turtles, Indian tent turtles, and Indian roofed turtles, taken from a first-class coach. Joint operation of the Strike Force, the Railway Protection Force, and the Bhopal Forest Division.
- July 2025, near Morena: 30 gharial hatchlings, 14 red-crowned roofed turtles, and in a follow-up raid 3 more red-crowned roofed turtles and 19 three-striped roofed turtles, believed to be heading for Bangladesh. The red-crowned roofed turtle’s remaining stronghold is the Chambal. It is not an animal that can be tipped into a village pond.

Prashant’s “every other week, 20 to 300 animals” is one specialist’s inbox. It matches the batch sizes in these cases. It is not a published national rate. He also said the rate at which animals then die from a bad release has never been compiled. That remains true.

### What a bad release looks like when someone did write it down

In January 2016, Dumka police seized more than 3,000 tortoises packed in gunny bags on a truck running from Uttar Pradesh toward West Bengal, and handed them to the forest department. The animals were released the same day into Massanjore dam. Within days, hundreds had washed up dead. Officials suspected they had already been suffocating in the truck. Nobody had identified them to a place they could survive, or held them long enough to find out. The deaths were reported. A proper count and a post-mortem were still pending when the news was filed. This is the shape of the gap: the seizure is counted, the aftermath is not.

In 2019–20 the Pune Forest Division held 63 turtles of five species, including spotted pond turtles and crowned river turtles, that had been smuggled west from Assam. They were kept by a herpetological society and a rescue trust because the local climate was a threat. In 2021, Turtle Survival Alliance India, with the Maharashtra and Assam forest departments, flew them back to Guwahati for quarantine at the state zoo before any return to a river. TSA’s stated reason is the same as Prashant’s: without that journey home, the animals are released into the wrong wetland or they stay in a cage for life. Of 461 Indian star tortoises seized at Thiruvananthapuram airport in 2014 and sent for rehabilitation inside their known range, 357 were still alive by the time they reached the centre. Infections and the stress of the seizure killed the rest, and the survivors needed a diet matched to that landscape before release.

None of these accounts is a mortality rate. Together they show that “we seized them” and “they lived” are different outcomes, and that the second one depends on a species name, an origin, and somewhere to keep them.

### Why a pocket card has not closed the gap

The 2023 cards cover 28 species and were written for exactly these officers. They already contain legal status, distribution, identification tips, and threats. Two limits remain.

The officer still has to match a dirty, stressed animal in a crate to a printed photograph, often dozens or hundreds of animals at a time, at a station. Half of recorded cases in the TRAFFIC decade still had no species name.

The cards also stop at identification tips. They do not tell a team how to hold 200 flapshells overnight, which river a Ganga turtle can go back to, or when an animal in the crate is the American red-eared slider rather than a protected native species. The slider is now established in urban lakes in several states, including Bhopal, Indore, and Jabalpur. Confusing it with a native Schedule I turtle, or the reverse, is treated in this project as the most common identification error in Indian casework.

A citizen-science app, Freshwater Turtles and Tortoises of India on the India Biodiversity Portal, already offers species pages and a place for enthusiasts to upload sightings. It is built to map where turtles are seen. It is not built for a seizure memo.

---



## What the app currently solves

The repository is further along than “an idea,” and shorter of the finish than “upload a photo and receive a name.” The reference library and the identification-by-eye tool work. The photograph tool is deliberately switched off.

### Ready for an officer who can look at the animal

**A species library of 30 kinds.** Sixteen pond and river turtles, eight softshells, five tortoises, and the red-eared slider. Sea turtles are outside the scope, because they are not what these consignments are. Each record has an English name, a Hindi name, the single mark that settles the identification, a short list of other marks, habitat, maximum size, a state-level map, a note for Madhya Pradesh, the species it is confused with and how to tell them apart, the IUCN category, the CITES appendix, and the 2022 schedule. Every record cites a handbook, a specialist-group account, or the 2023 cards. The schedules were checked against the Zoological Survey of India’s reptile checklist and against those cards. The app says, on the page, that this is a field hint. A court paper still has to be checked against the Gazette.

**A key that does not require a photograph or a model.** The officer ticks whatever can actually be seen — shell, belly, head, feet — in any order. The tool ranks the species that fit. If a mark contradicts a species, that species is pushed down and kept on the list, because a mark misread in bad light should not delete the right answer.

**A distribution answer.** For each species the app shows the states where it is recorded and the kind of place it lives. If the recorded state is outside that range, the design treats that as a clue that the animal was moved by trade, not as a reason to pretend the species cannot be there. That is the right instinct for a seizure. An animal from Assam found in Pune is the case, not an error. > **Ajinkya:** Potentially a module that we would want to build.

**A photograph recipe.** Five frames: the back shell from directly above, the belly shell flat and whole, a side view, the head close up in shade, and something for scale. The belly view is the one that separates the two roofed turtles on different schedules. Direct sun washes out the marks. This is written for field staff because most failed identifications are failed photographs. Photograph guidance experience that captures the appropriate photos.

**A log.** Each time the tool is used, it stores who looked, where, and what it concluded. Ambiguous cases are kept, because those are the photographs worth adding later.

### Not ready, and honestly labelled as such

**Upload a picture and receive a species.** Nothing in the deployed app is allowed to do this yet. A first attempt at matching photos against the printed identification cards scored worse than guessing: right none of the time, against about a four percent chance of naming a species at random. The reason is ordinary. A scanned plate from a book looks more like other scanned plates than like a phone photo of a turtle in a crate. The app refuses to put a percentage on that kind of match, because a confident wrong name on a seizure memo is worse than no name. -> **Ajinkya:**  A combination of AI and continuous training aan image analysis model would help in getting to the highest accuracy here? I need technical assistance here and perhaps a research AI agent that continously guides that AI?

The path that would turn the photograph tab on is the one Prashant described: real photographs of different individual animals, both shell surfaces, checked by someone who knows the species, and only then a model that is also allowed to say “I don’t know.” Rare species with only a handful of photos should produce a referral to a specialist, not a guess. The machinery for accepting photographs, stripping location data out of them, and reviewing them before they teach the tool is built. > **Ajinkya:**  The photographs are not. The matching experience should accomodate this?

**Husbandry.** The original brief asked for care as well as a name and a map. The app has habitat and size, which tell you the kind of place an animal belongs. It does not have care sheets: water depth, temperature, food, how long to quarantine, or a rule for which river a recovered animal may re-enter. That part of the idea is still ahead. > **Ajinkya:**  Not sure what we can do here? Ahead means it has been figured out if not lets figure it out then.

**A count of turtles in the landscape.** Out of scope, by design. The users are people holding confiscated animals.

---



## The paper Prashant sent

[SOCI-YOLO11n](https://www.sciencedirect.com/science/article/pii/S1574954126001688) (Wang and colleagues, *Ecological Informatics*, 2026) is a camera tool for Hainan, China. It watches for invasive freshwater turtles, especially the red-eared slider, in cluttered outdoor scenes, across seven species. On their test images it found the animals reliably and is small enough to run on a small field computer. The problem it solves is “is there a turtle in this frame, and can we run that check all day on a riverbank camera.”

The problem in this project is “which of thirty Indian species is in this officer’s photograph, and what should happen to it next.” The Hainan work is evidence that this family of image tools can pick turtles out of a messy background. It is not a dataset, a species list, or a legal reference for Indian seizures. A model trained on Hainan cameras would not know an Indian roofed turtle from a brown roofed turtle.

Two closer studies, both from Thailand, show what species-level identification actually required:

- Chaianunporn and colleagues (*Scientific Reports*, 2024) separated three snail-eating turtles that are hard to tell apart, because Thai law protects them differently. They used several views of each animal, including the face and the belly shell, and they checked the image result against DNA. One view was not enough. Several views, of the marks that actually differ, were.
- A follow-on study (*PeerJ Computer Science*, 2025) used belly-shell stripes and facial stripes to recognise individual Khorat snail-eating turtles. That is “which animal,” which is a harder trick than “which species,” and it again depended on those patterns being photographed on purpose.

A drone study on the Ganga (2024) could tell a turtle from a gharial in aerial photos. It stopped at “turtle.” That is the same limit as a generic image tool: it can see that the animal is a turtle. The legal and the release decisions need the species, and the species is in the belly colour, the keel on the shell, and the marks on the head.

The same limit answers the question of whether one photo tool could later do elephants, or any other animal, from the same model. Finding “an animal in a picture” transfers. Naming the species does not. Elephants are identified by ears, tusks, and other traits that turtles do not have. Indian turtles are identified by shell and head marks that an elephant model has never seen. Each group needs photographs of its own animals, labelled by people who can already tell them apart, before a tool should be trusted with a name.

---



## Where that leaves the work


| What an officer needs at a seizure                        | Where it stands                                           |
| --------------------------------------------------------- | --------------------------------------------------------- |
| A name, from marks they can see on the animal             | Usable now, through the key and the species cards         |
| Where that species lives, by state and habitat            | Usable now                                                |
| Which schedule to write on the memo, as a field hint      | Usable now, with an explicit warning to check the Gazette |
| A name from a photograph alone                            | Switched off until real, checked photographs exist        |
| How to keep the batch alive, and where it may be released | Habitat only. Care and release rules are not written      |
| A running record of what was decided                      | Usable now, on the machine that runs the app              |


The published gap is the one TRAFFIC measured in 2019: in about half of seizure cases, nobody recorded the species. The local gap is the one Prashant is filling by hand: a specialist looking at photographs of 20 to 300 animals, every other week, for teams whose next decision is a charge and a release. The app currently replaces the pocket card with a searchable key and a distribution note. It does not yet replace the specialist, and it does not yet say where to put the animal back.

---



## Sources

**The project**

- This repository, especially the readme and the species reference (`data/species_db.json`), compiled 2026 from Das (1995), Ahmed and Das (2010), Moll (1987), the IUCN Tortoise and Freshwater Turtle Specialist Group accounts, Rhodin and colleagues (*Turtles of the World*, 9th edition, 2021), and the 2023 identification cards.
- Singh, S., Badola, S. and Fernandes, M. (2023). *Identification Cards: Tortoises and Freshwater Turtles of India.* TRAFFIC India, TSA-India, and WWF-India. Updated for the Wild Life (Protection) Amendment Act, 2022.
- Mohapatra, P.P. and colleagues (2024). *Fauna of India Checklist: Reptilia*, version 1.0. Zoological Survey of India. Schedule lists used here follow their Table 4.
- Wild Life (Protection) Amendment Act, 2022. Field schedules in the app are a triage aid. The Gazette is the legal text.

**Seizures and the identification gap**

- Mendiratta, U., Sheel, V. and Singh, S. (2017). Enforcement seizures reveal large-scale illegal trade in India’s tortoises and freshwater turtles. *Biological Conservation* 207: 100–105. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S000632071730160X). 223 reports, 2011–2015, at least 58,442 live animals, at least 15 species.
- Badola, S., Choudhary, A.N. and Chhabra, D.B. (2019). *Tortoises and Freshwater Turtles in illegal trade in India.* TRAFFIC India, New Delhi. Factsheet: [TRAFFIC PDF](https://www.traffic.org/site/assets/files/12464/traffic-factsheet-tortoises-freshwater-turtles-india-2019.pdf). Minimum 111,312 animals, September 2009–September 2019; species unreported in 51.5 percent of cases.
- TRAFFIC (11 October 2019, cards updated May 2023). [New ID tools to help curb illegal trade of tortoises and freshwater turtles in India](https://www.traffic.org/news/new-id-tools-to-help-curb-illegal-trade-of-tortoises-and-freshwater-turtles-in-india-launched/).
- Sengottuvel, R.R., Mendis, A., Sultan, N., Shukla, S., Chaudhuri, A. and Mendiratta, U. (2024). From pets to plates: network analysis of trafficking in tortoises and freshwater turtles representing different types of demand. *Oryx* 58(1): 78–89. Cited here for the pets-versus-meat split; the case counts quoted above are as reported by Mandal and colleagues.
- Mandal, P., Halder Mallick, P. and Bhattacharya, T. (2024). An analysis of illegal trade of non-marine turtles in West Bengal, India: study based on a seven-year confiscation scenario. Research Square preprint, doi:10.21203/rs.3.rs-4812000/v1. 133 reports, 2017–2023, 33,317 animals. Includes the warning on release into local water bodies.

**After the seizure**

- Indian Express, 16 January 2016. [Jharkhand: rescued tortoises found dead days after release in a dam](https://indianexpress.com/article/india/india-news-india/jharkhand-rescued-tortoises-found-dead-days-after-release-in-dam/). More than 3,000 animals, Dumka, released the same day into Massanjore dam.
- The Sentinel / EastMojo, August 2021, reporting Turtle Survival Alliance India. Sixty-three turtles of five species, seized in Pune in 2019–20, flown back to Assam for quarantine. TSA: without repatriation they are released into an unsuitable wetland or remain captive for life.
- Wildlife Trust of India. [The stars are wild again](https://old.wti.org.in/news/the-stars-are-wild-again/). 461 Indian star tortoises seized at Thiruvananthapuram airport, 30 April 2014; 357 alive on arrival at a rehabilitation centre inside their range.
- Times of India, 2025, on the 30 September 2023 Itarsi seizure. 282 Indian tent turtles; 280 released into the Tawa reservoir. [Court report](https://timesofindia.indiatimes.com/city/bhopal/court-sentences-4-in-turtle-smuggling-case/articleshow/122460851.cms).
- Indian Express, 2026. [Inside the railway network moving India’s protected turtles](https://indianexpress.com/article/india/express-special-from-ganga-to-global-pet-markets-police-in-madhya-pradesh-map-rail-route-of-protected-turtles-10760782/). Strike Force and railway police: 647 turtles from trains in Madhya Pradesh, 2020–2026.
- NDTV / News18, February 2026. 311 turtles from the Patna–Indore Express at Bhopal.
- Times of India, July 2025. Strike Force seizure near Morena: gharial hatchlings and red-crowned and three-striped roofed turtles believed bound for Bangladesh.

**Image identification, for context**

- Wang, X. and colleagues (2026). SOCI-YOLO11n: a deep learning model for identifying invasive freshwater turtles in complex ecological contexts. *Ecological Informatics.* [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1574954126001688). Seven species, Hainan, camera detection of invasive turtles. The paper Prashant sent.
- Chaianunporn, T. and colleagues (2024). Automated classification in turtles genus *Malayemys* using ensemble multiview images based on improved YOLOv8 with CNN. *Scientific Reports.* Species identification for enforcement, checked against DNA.
- PeerJ Computer Science (2025). Individual recognition of the Khorat snail-eating turtle from facial and belly-shell patterns.
- ISPRS Archives (2024). Drone imagery along the Ganga: a turtle detected as a turtle, separately from gharials. Not a species identification.

**Conversation**

- Prashant Jijaji and Ajinkya Patil, 8 October 2026. Use case, users, batch size, and the limits of what is known about post-release deaths. Those figures are his working experience, not a published census.

