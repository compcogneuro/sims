*[Back to simulations](https://compcogneuro.org/simulations)*

# Introduction

This network learns to encode both syntax and semantics of sentences in an integrated "gestalt" hidden layer. The sentences have simple agent-verb-patient structure with optional prepositional or adverb modifier phrase at the end, and can be either in the active or passive form (80% active, 20% passive). There are ambiguous terms that need to be resolved via context, showing a key interaction between syntax and semantics.

The original sentence gestalt model was published by [St. John & McClelland (1990)](#references), and this version uses the same language corpus but a somewhat different network architecture and training paradigm.

This version of the model uses the more recent *DeepLeabra* framework [(O'Reilly et al, 2021)](#references) that incorporates the functions of the deep layers of the neocortex and its interconnectivity with the thalamus.  We hypothesize that these circuits support a powerful form of predictive error-driven learning, where the top-down cortical projections drive a prediction on the pulvinar nucleus of the thalamus, which serves as the minus phase, and then phasic bursting inputs drive a periodic plus phase representing what actually happened.  This bursting happens at the 10 Hz *alpha* frequency (every 100 msec), and forms the basis for the use of this time scale as the primary organizing principle in our models.

The current model uses these circuits to learn by attempting to predict which word will appear next in the sentences -- this prediction is represented in the `EncodeP` pulvinar layer.  To support predictive learning, the deep layers also have the ability to maintain "context" information from the previous time step, and this is also critical for enabling the model to remember information from prior words in the sentence.  This contextual information is similar to the simple recurrent network (SRN) model [Elman, 1990](#references), which was used in the prior version of this simulation.  Interestingly, we found that although it is difficult for the model to accurately predict the next word, learning to do so nevertheless benefitted overall learning on answering Role-Filler questions as described below.

In the many years since the original sentence gestalt model, and since this replication of the model in 2000, increasingly large "AI" models have been developed based on the same word prediction learning process. As you probably are aware, these GPT (_generative pretrained transformer_) models have developed fairly amazing levels of understanding about all manner of human knowledge by "reading" and predicting the next word in a vast corpus of text that encompasses a significant fraction of everything that humans have written over the years. Future editions of this textbook will explore this progression and the transformer technology that powers the GPT models.

# Training

In addition to the predictive learning, the network is trained by asking questions about current and previous information presented to the network.  During the minus phase of every trial, the current word is on the `Input`, and a question is posed in the `Role` input layer about what a particular semantic role is for the current sentence. For example, for the first word in an active sentence, which is always the "agent" of the sentence, the `Role` input will be `Agent`, and the network just needs to produce on the `Filler` output layer the same thing that is present in the `Input`. However, on the next trial, a new word (the verb or `Action`) is presented, but the network must still remember the `Agent` from the prior trial, as queried by a "review" question. This is what the `GestaltCT` corticothalamic context layer provides. You can see all of this in the testing we'll perform next. If you want to see the training process in more detail, you can click on [[sim:Init]] and [[sim:Step]] `Trial` through a few trials.

The `curq` question type indicates a question about the current filler, which the model almost always gets correct, whereas `revq` indicates a "review" question about something earlier in the sentence -- these are the most revealing about the model's internal "mental state".  There are 2 systematic review questions during the Action and Patient inputs (which ask about the prior Agent and Action inputs), and then two random review questions, one during the syntactic prepositional word (e.g., `with`, `to`), and a final one at the very end.  The previous version and the original model both asked about everything after every word input, but we are able to streamline things here and still get good encoding, in part due to the nature of the deep leabra learning mechanism.

* Next, press [[sim:Open Trained Wts]] in the toolbar to load pre-trained weights (the network takes many minutes to hours to train). Then, you can start by poking around the network and exploring the connectivity using the [[sim:Wts]] -> `r.Wt` view, and then return to viewing [[sim:Act]] -> `Act`.

# Testing

Now, let's evaluate the trained network's performance, by exploring its performance on a set of specially selected test sentences listed here:

* Role assignment:
    + Active semantic: The schoolgirl stirred the Kool-Aid with a spoon.
    + Active syntactic: The busdriver gave the rose to the teacher.
    + Passive semantic: The jelly was spread by the busdriver with the knife.
    + Passive syntactic: The teacher was kissed by the busdriver.
    + Active control: The busdriver kissed the teacher.
*  Word ambiguity:
    + The busdriver threw the ball in the park.
    + The teacher threw the ball in the living room.
* Concept instantiation:
    + The teacher kissed someone (male).
* Role elaboration
    + The busdriver ate steak (with knife).
    + The pitcher ate something (with spoon).
* Online update
    + The child ate soup with daintiness.
    + control: The pitcher ate soup with daintiness.
* Conflict
    + The adult drank iced tea in the kitchen (living room).

The test sentences are designed to illustrate different aspects of the sentence comprehension task, as noted in the table. First, a set of role assignment tasks provide either semantic or purely syntactic cues to assign the main roles in the sentence. The semantic cues depend on the fact that only animate nouns can be agents, whereas inanimate nouns can only be patients. Although animacy is not explicitly provided in the input, the training environment enforces this constraint. Thus, when a sentence starts off with "The jelly...," we can tell that because jelly is inanimate, it must be the patient of a passive sentence, and not the agent of an active one. However, if the sentence begins with "The busdriver...," we do not know if the busdriver is an agent or a patient, and we thus have to wait to see if the syntactic cue of the word *was* appears next.

* Set the run mode to `Test` instead of `Train`, and do [[sim:Init]] and then [[sim:Step]] `Trial` to see the start of the first test sentence, which is the special `start` indicator. Do `Step Trial` again to see the first word in the sentence.

The first word of the active semantic role assignment sentence (`schoolgirl`) is presented, and the network correctly answers that schoolgirl is the agent of the sentence, which is shown in the `Output` in the text at the bottom of the network.  The `TrlErr` output shows whether an error was made relative to the target.

Note that there is no plus-phase and no training during this testing, so everything depends on the integration of the input words.

* Continue to do [[sim:Step]] `Trial` through to the final word in this Active semantic sentence (spoon). You can click on the [[sim:Test Trial Plot]] tab to see a bar-plot of each trial as it goes through, with the network's [[sim:Output]] response. It is probably easier to see the full results in the [[sim:Test Trial]] tab, which shows the [[sim:SentType]] indicating the type of test sentence.

You should observe that the network is able to correctly identify most of the roles of the words presented, and it makes a generally sensible response when it makes a mistake. Because in this sentence the roles of the words are constrained by their semantics, this success demonstrates that the network is sensitive to these semantic constraints and can use them in parsing.

* [[sim:Step]] `Trial` through the next sentence (Active syntactic).

This sentence has two animate nouns (busdriver and teacher), so the network must use the syntactic word order cues to infer that the busdriver is the agent, while using the "gave to" syntactic construction to recognize that the teacher is the recipient. Observe that through the multiple queries of Agent, it remembers correctly that it is the previously seen busdriver, not the more recently seen teacher (although it might make a mistake sometimes).

In the next sentence, the passive construction is used, but this should be obvious from the semantic cue that `jelly` cannot be an agent.

* [[sim:Step]] through Passive semantic and observe that the network correctly parses this sentence.

In the final role assignment case, the sentence is passive and there are only syntactic constraints available to identify whether the teacher is the agent or the patient. This is one of the most difficult constructions that the network faces, and it sometimes makes errors on it.

* [[sim:Step]] through this Passive syntactic sentence.

Further testing has shown that the network sometimes gets this sentence right, but often makes errors. This can apparently be attributed to the lower frequency of passive sentences, as you can see from the next sentence, which is a "control condition" of the higher frequency active form of the previous sentence, with which the network has no difficulties (although occasionally it makes a random error here and there).

* [[sim:Step]] through this Active control sentence.

The next two sentences test the network's ability to resolve ambiguous words, in this case *throw* and *ball* based on the surrounding semantic context. During training, the network learns that busdrivers throw baseballs, whereas teachers throw parties. Thus, the network should produce the appropriate interpretation of these ambiguous sentences.

* [[sim:Step]] through these next two sentences (Ambiguity1 and 2) to verify that this is the case.

Sometimes we have observed the network making a mistake here by replacing *teacher* with the other agent that also throws parties, the *schoolgirl*. Thus, the network's context memory is not perfect, but it tends to make semantically appropriate errors, just as people do. 

The next test sentence probes the ability of the network to instantiate an ambiguous term (e.g., *someone*) with a more concrete concept. Because the teacher only kisses males (the pitcher or the busdriver), the network should be able to instantiate the ambiguous *someone* with either of these two males.

* As you [[sim:Step]] through this sentence, observe that `someone` is instantiated with `PitcherPers`. 

A similar phenomenon can be found in the role elaboration test questions. Here, the network is able to answer questions about aspects of an event that were not actually stated in the input. For example, the network can infer that the schoolgirl would eat crackers with her fingers.

* Step through the next sentence (Role elaboration1).

You should see that the very last question regarding the instrument role is answered correctly with fingers, even though fingers was never presented in the input. The next sentence takes this one step further and has the network try to guess what the pitcher might eat.

* Go ahead and [[sim:Step]] through this one (Role elaboration2).

Interestingly, it the next word predictor suggests soup, but the Filler output is "finger"!  ouch.  It then does correctly infer that the patient is none other than the pitcher, which would be true if he was eating his own finger..  Anyway, the model is sometimes a bit random.

The next test sentence is intended to evaluate the online updating of information in a case where subsequent information further constrains an initially vague word. In this case, the sentence starts with the word *child*, which the model guesses is `pitcher`, likely left over from the previous sentence context.

* [[sim:Step]] through the Online Update sentence.

When the network gets to the adverb `daintiness`, this should uniquely identify the `schoolgirl`, but the model instead fills in `teacher`, which is the other person who eats with daintiness, but they are not a child.

To compare the effect that *daintiness* is having, we can run the next control condition where the pitcher is explicitly specified as the agent of the sentence -- the model gets fairly confused at one point and repeats its confusion about the teacher being the agent.

* [[sim:Step]] through the Online control sentence.

The final test sentence illustrates how the network deals with conflicting information. In this case, the training environment always specifies that iced tea is drunk in the living room, but the input sentence says it was drunk in the kitchen. 

* [[sim:Step]] through this Conflict sentence.

Although it accepts the kitchen location, it then states that the action must have been `spread` instead of `drank`, so it has updated its schema based on the conflicting input to an action that happens in the kitchen.

This may provide a useful demonstration of how prior knowledge biases sentence comprehension, as has been shown in the classic "war of the ghosts" experiment (Bartlett, 1932) and many others.

# Nature of Representations

Having seen that the network behaves reasonably (if not perfectly), we can explore the nature of its internal representations (in the Gestalt layer) to get a sense of how it works.

* Select the [[sim:NounClust]] tab, and press [[sim:Probe all]] in the toolbar.

After a short delay, the cluster plot for the nouns will show up.  This cluster plot shows some sensible pairings of words such as "schoolgirl" and "daintiness" and "crackers" and "jelly".  However, these results are limited because the words were just presented outside of any sentence context, which the model never experienced.

* Click the [[sim:SentClust]] tab to see the sentence probe cluster plot.

These probe sentences systematically vary the agents, verbs, and patients to reveal how the Gestalt layer encodes these factors.  Agents: `bu` = busdriver, `te` = teacher, `pi` = pitcher, `sc` = schoolgirl; Verbs: `dr` = drank, `st` = stirred, `at` = ate; Patients: `ic` = icedtea, `ko` = koolaid, `so` = soup,  `st` = steak.

This cluster plot shows that the sentences are first clustered together according to verb, and when the verb is the same (ate), the patient (soup vs. steak) breaks apart the clusters.  Within each verb cluster, the grouping of the agents also is organized systematically -- busdriver and schoolgirl are the most different and they never group, whereas busdriver, teacher and pitcher, schoolgirl are most typically related. Thus, we can see that the gestalt representation encodes information in a systematic fashion, as we would expect from the network's behavior. 

> **Question 10.12:** Does this cluster structure reflect purely syntactic information, purely semantic information, or a combination of both types of information? Try to articulate in your own words why this kind of representation would be useful for processing language.

<sim-question id="10.12">

# References

* Elman, J. L. (1990). Finding Structure In Time. Cognitive Science, 14(2), 179–211.

* O’Reilly, R. C., Russin, J. L., Zolfaghar, M., & Rohrlich, J. (2020). Deep Predictive Learning in Neocortex and Pulvinar. Journal of Cognitive Neuroscience, 33(6), 1158–1196. [PDF](https://ccnlab.org/papers/OReillyRussinZolfagharEtAl21.pdf)

* St John, M. F., & McClelland, J. L. (1990). Learning and Applying Contextual Constraints in Sentence Comprehension. Artificial Intelligence, 46, 217–257.

