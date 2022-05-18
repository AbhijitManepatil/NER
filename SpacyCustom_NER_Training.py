def customizing_pipeline_component(nlp: Language,train_data ):
    # NOTE: Starting from Spacy 3.0, training via Python API was changed. For information see - https://spacy.io/usage/v3#migrating-training-python
#     train_data = [
#         ('We need to deliver it to Festy.', [(25, 30, 'DISTRICT')]),
#         ('I like red oranges', [])
#     ]

    # Result before training
    print(f"\nResult BEFORE training:")
    data="""ECHNICAL STRENGTHS Computer Language Java/J2EE, Swift, HTML, Shell script, MySQL Databases MySQL Tools SVN, Jenkins, Hudson, Weblogic12c Software Android Studio, Eclipse, Oracle, Xcode Operating Systems Win 10, Mac (High Sierra) Education Details 
June 2016 B.E. Information Technology Goregaon, MAHARASHTRA, IN Vidyalankar Institute of Technology
May 2013   Mumbai, Maharashtra Thakur Polytechnic
May 2010   Mumbai, Maharashtra St. John's Universal School
Java developer """
    doc = nlp(data)
    print_doc_entities(doc)

    # Disable all pipe components except 'ner'
    disabled_pipes = []
    for pipe_name in nlp.pipe_names:
        if pipe_name != 'ner':
            nlp.disable_pipes(pipe_name)
            disabled_pipes.append(pipe_name)

    print("   Training ...")
    optimizer = nlp.create_optimizer()
    for _ in range(25):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            example = Example.from_dict(doc,{"entities": entity_offsets})
            nlp.update([example], sgd=optimizer)

    # Enable all previously disabled pipe components
    for pipe_name in disabled_pipes:
        nlp.enable_pipe(pipe_name)

    # Result after training
    print(f"Result AFTER training:")
    doc = nlp(data)
    print_doc_entities(doc)
