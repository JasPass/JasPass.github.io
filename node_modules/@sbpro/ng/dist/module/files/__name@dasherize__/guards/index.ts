import { <%= classify(name) %>Guard } from './<%= dasherize(name) %>.guard';

export const guards = [<%= classify(name) %>Guard];

export * from './<%= dasherize(name) %>.guard';
