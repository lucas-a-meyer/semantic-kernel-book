﻿using System.Runtime.InteropServices;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.SemanticFunctions;

var builder = new KernelBuilder();

var (apiKey, orgId) = Settings.LoadFromFile();

builder.WithOpenAIChatCompletionService("gpt-3.5-turbo", apiKey, orgId, serviceId: "gpt3", setAsDefault: true);
builder.WithOpenAIChatCompletionService("gpt-4", apiKey, orgId, serviceId: "gpt4", setAsDefault: false);

IKernel kernel = builder.Build();

string prompt = "Finish the following knock-knock joke. Knock, knock. Who's there? {{$input}}, {{$input}} who?";
var jokeFunction = kernel.CreateSemanticFunction(prompt, temperature: 0.8);

var showManagerPlugin = kernel.ImportSkill(new Plugins.ShowManager());

var result = await kernel.RunAsync(showManagerPlugin["RandomTheme"]);
Console.WriteLine("I will tell a joke about " + result);

var joke = await jokeFunction.InvokeAsync(result);
Console.WriteLine(joke);
